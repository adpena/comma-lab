---
council_tier: T1
council_attendees: [Shannon, Dykstra, AssumptionAdversary, NSCS06-v7-author-cite]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Path (b) JUSTIFIED-EXTEND-WITH-FORK on 4 cargo-cults preserves substrate-optimal engineering per Catalog #290"
    classification: HARD-EARNED
    rationale: "Phase 1 cargo-cult audit produced a per-layer canonical-vs-unique decision table with explicit FORK rationales on cargo-cults #3, #5, #8, #12. The 4 FORK decisions are each backed by an empirical or first-principles justification; the 8 ADOPT/PRESERVE decisions are each backed by HARD-EARNED prior empirical work."
council_decisions_recorded:
  - "op-routable #1: Phase 3 L0/L1 SCAFFOLD implements the 4 FORK paths via new mlx_iteration_v2 module + cls_stream consumption at L0 inflate + hash-derived seed alternative + Catalog #324 axis-labeling discipline"
  - "op-routable #2: Phase 3 design memo MUST cite each FORK rationale per Catalog #290 + each cargo-cult unwind path per Catalog #303"
  - "op-routable #3 (operator-routable; deferred-pending-symposium): per-substrate symposium per Catalog #325 gates first paid dispatch; Phase 3 lands the BUILD that the symposium will adjudicate but does NOT trigger dispatch"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
horizon_class: frontier_pursuit
---

# Path 3 C' — NSCS06 v8 chroma_lut substrate-design decision memo

**Date:** 2026-05-26
**Lane:** `lane_path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526`
**Phase:** 2 (substrate-design decision based on Phase 1 cargo-cult audit)
**Phase 1 audit:** `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md` (commit `a6e2a06e3`)

---

## Decision

**Path (b) JUSTIFIED-EXTEND-WITH-FORK** per Phase 1 cargo-cult audit Section "Phase 2 decision input".

The existing `src/tac/substrates/nscs06_v8_chroma_lut/` scaffold (3685 LOC + 1658 LOC tests + predecessor 762 LOC MLX iteration module + 1007 LOC trainer) carries significant accreted engineering work that:

- HARD-EARNED layers (architecture LUT shape per Assumption #1, archive grammar per #9, 6-DOF affine warp per #6, infrastructure-canonical-adoption per #7, etc.) are PRESERVED.
- CARGO-CULTED-CRITICAL layers (uniform-PRNG distribution per #3, L0 cls=0 uniform per #5, SegNet noise-floor probe per #8, rate-axis-only predicted band per #12) are FORKED with explicit canonical-vs-unique decision rationale per Catalog #290.

Path (a) JUSTIFIED-EXTEND with canonical adoption was REJECTED in Phase 1 because 4 critical cargo-cults remain.
Path (c) FRESH SUBSTRATE DESIGN from first principles was REJECTED because the cargo-culted assumptions are LOCALIZED to 4 surfaces and can be FORKED without re-architecting.

---

## Binding implementation roadmap for Phase 3

### 3a. NEW canonical helper: per-class chroma byte-mutation distinguishing-feature smoke

| | |
|---|---|
| **New file** | `src/tac/substrates/nscs06_v8_chroma_lut/distinguishing_feature_smoke.py` (~150 LOC) |
| **API** | `verify_per_class_chroma_anchors_consumed_at_inflate(archive_bytes, output_dir, classes_to_mutate=(1,2,3,4)) -> PerClassChromaDistinguishingFeatureVerdict` |
| **Operational contract** | For each class c in classes_to_mutate, mutate LUT[:, c, :] bytes in archive_bytes, run inflate, compare frame-1 RGB bytes vs unmutated baseline. Returns `PASS` (rendered frames change for each class c) or `FAIL_AT_CLASS_<c>` (rendered frames identical → distinguishing feature NOT honored). |
| **Catalog mapping** | Catalog #272 distinguishing-feature integration contract + Catalog #297 reversibility audit + Catalog #220 substrate L1+ operational mechanism |
| **Unwinds cargo-cult** | #5 (L0 SCAFFOLD class=0 uniform structural test-invalidity) |
| **Acceptance test** | Phase 3 tests fixture pass at v1 inline-LUT path WITH cls_stream consumed; fixture fails at L0 cls=0 uniform path with explicit `# CHROMA_DISTINGUISHING_FEATURE_TEST_INVALID_AT_L0:cargo_cult_5_unwind_required_for_L1_promotion_per_path_3_c_prime_audit_landed_20260526` waiver until cls_stream wired |

### 3b. NEW canonical helper: GT-distribution-matched seed-derivation alternative arm

| | |
|---|---|
| **New file** | `src/tac/substrates/nscs06_v8_chroma_lut/gt_distribution_matched_seed.py` (~120 LOC) |
| **API** | `derive_chroma_lut_seed_from_gt_lut_bytes(chroma_lut_bytes: bytes, seed_size: int = 32, kind: str = 'sha256_truncated') -> bytes` (canonical hash-derived alternative to PCG64-uniform) |
| **Operational contract** | The seed bytes ARE a deterministic function of the GT-derived LUT bytes (sha256 truncated to seed_size). At inflate, the LUT is reconstructed via the canonical `derive_codebook_from_seed` helper using PCG64 keyed by the GT-derived seed. The seed bytes change WITH the GT distribution; PCG64 still produces uniform-looking bytes BUT the inflate-reconstructed LUT is now derived FROM-A-SEED-DERIVED-FROM-GT (i.e. the seed encodes the GT-distribution-fingerprint). The empirical question is whether this 32-byte fingerprint is sufficient to recover the GT chroma distribution within the SegNet noise-floor tolerance. |
| **Catalog mapping** | Catalog #290 canonical-vs-unique decision per layer FORK |
| **Unwinds cargo-cult** | #3 (PCG64-uniform LUT distribution vs GT chroma distribution mismatch) |
| **Honest disclosure** | This is an ALTERNATIVE arm, NOT a guaranteed unwind. The empirical question is whether GT-fingerprint-keyed PCG64 produces a distinguishing-from-PCG64-uniform inflate output. If the empirical answer is "no", the unwind path is FALSIFIED-AT-IMPLEMENTATION-LEVEL per Catalog #307 (paradigm-vs-implementation classification) and the cargo-cult-unwind moves to PATH 2: replace PCG64 with a chroma-distribution-aware codebook generator. |
| **Acceptance test** | Phase 3 test: derive seed from synthetic GT-LUT-bytes; verify seed bytes are deterministic + length=32; verify inflate-reconstructed LUT is NOT byte-identical to PCG64-uniform-from-random-seed baseline. |

### 3c. NEW canonical helper: $0 MLX-local CHEAP-SIGNAL-FIRST SegNet noise-floor probe

| | |
|---|---|
| **Extension to** | `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (predecessor module; ~80 LOC added) |
| **API** | `measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline(rgb_pairs_gt, chroma_lut_v8, mlx_segnet_adapter, baseline='gt_rgb') -> SegNetArgmaxDisplacementVerdict` |
| **Operational contract** | Render frame-1 RGB from v8 LUT at compress time; run MLX SegNet; measure argmax-flip-fraction vs SegNet-on-GT-RGB-baseline. Returns `(displacement_fraction, in_noise_floor: bool, recommended_proceed: bool)`. Threshold: `displacement_fraction < 1e-3` → in noise floor → v8 distinguishing feature below SegNet sensitivity → recommend `NO_GO_PROCEED_TO_PATH_2_REDESIGN`. |
| **Catalog mapping** | Catalog #297 reversibility audit + Catalog #287 empirical-claim-without-evidence-tag |
| **Unwinds cargo-cult** | #8 (SegNet noise-floor sensitivity to LUT chroma differentiation UNKNOWN) |
| **Cost** | $0 (Apple Silicon MLX; runs in <2 minutes on M5 Max per predecessor MLX iteration baseline) |
| **Acceptance test** | Phase 3 test: synthetic GT pairs + synthetic chroma LUT; verify displacement_fraction is finite + verdict object is constructable. Real-MLX integration test deferred to operator-routable (requires `mlx.core` import). |

### 3d. NEW canonical helper: Catalog #324-grade predicted-band axis-labeling discipline

| | |
|---|---|
| **Extension to** | `src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py` predicted-band declaration |
| **API** | `predicted_delta_s_with_axis_attribution() -> dict[str, float]` returning `{"rate_axis": -0.002706, "seg_axis": "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324", "pose_axis": "UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324"}` |
| **Sister recipe field** | Phase 3 documents (but does NOT modify) the canonical operator-authorize recipe YAML field `predicted_band_validation_status: pending_post_training` per Catalog #324 |
| **Catalog mapping** | Catalog #324 (predicted-band-from-incomplete-Tier-C self-protect anchor) |
| **Unwinds cargo-cult** | #12 (rate-axis-only predicted band axis-labeling discipline) |
| **Acceptance test** | Phase 3 test: returned dict has `rate_axis` numeric + `seg_axis` + `pose_axis` strings carrying the canonical `UNKNOWN_PENDING_PAIRED_SMOKE_PER_CATALOG_324` token |

### 3e. EXTEND existing test_substrate.py + test_mlx_iteration.py with cargo-cult-unwind verification tests

| | |
|---|---|
| **Extension** | Add ~10 tests covering: (i) per-class distinguishing-feature smoke at L0 (expected: FAIL_AT_CLASS_1 because cls=0 uniform); (ii) gt-distribution-matched seed derivation determinism + length; (iii) SegNet noise-floor probe constructability; (iv) predicted_delta_s_with_axis_attribution returns canonical UNKNOWN tokens; (v) regression test that predecessor's 31 MLX iteration tests still pass. |
| **Catalog mapping** | Catalog #229 PV (predecessor work preserved) + Catalog #110/#113 APPEND-ONLY (test additions only; no mutation of existing tests) |

### 3f. NEW Phase 3 design memo with full 8-section discipline

Per Phase 3 of methodology + Catalog #294 9-dim checklist + Catalog #300 v2 frontmatter + Catalog #305 observability + Catalog #309 horizon_class + Catalog #325 per-substrate symposium contract.

Required sections:
- `## Canonical-vs-unique decision per layer` (Catalog #290) — 4 FORK rationales tabled
- `## 9-dimension success checklist evidence` (Catalog #294) — per-dimension evidence
- `## Cargo-cult audit per assumption` (Catalog #303) — cite Phase 1 audit memo + 4 unwind path additions
- `## Observability surface` (Catalog #305) — 6 facets including byte-mutation traceability for each class chroma anchor
- `## Predicted ΔS band` (Catalog #296 Dykstra feasibility + Catalog #324 axis-labeling) — explicit AXIS-DECOMPOSED prediction with UNKNOWN tags on seg+pose
- `horizon_class: frontier_pursuit` declaration (Catalog #309)
- Chroma_lut substitution math sub-section with explicit canonical equation #26 IN-DOMAIN context citation + Catalog #359 anti-misapplication self-protect
- MLX-implementation roadmap sub-section

### 3g. Lane registry pre-registration per Catalog #126

```bash
.venv/bin/python tools/lane_maturity.py add-lane \
    lane_path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526 \
    --name "Path 3 C' NSCS06 v8 chroma_lut cargo-cult-first L0/L1 SCAFFOLD" \
    --phase 3 \
    --notes "research_only=true; cargo-cult-pass-first methodology per directive #2 2026-05-26; sister to lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521 (existing scaffold preserved); FORK on cargo-cults #3 #5 #8 #12 per Phase 1 audit a6e2a06e3"
```

Initial gates marked at L0:
- `impl_complete` (Phase 3 L0 landing)
- `strict_preflight` (after gate validation)
- `memory_entry` (Phase 3 landing memo)

Pending gates (per Catalog #325 per-substrate symposium):
- `real_archive_empirical` (first paired smoke — operator-gated)
- `contest_cuda` (first contest-CUDA anchor — operator-gated)
- `three_clean_review` (R1-R3 council rounds)
- `deploy_runbook` (driver script already lands at sister lane; promotion requires per-substrate symposium PROCEED + paired smoke success)

---

## Constraint inheritance from CLAUDE.md non-negotiables

- **Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY**: Phase 3 L0/L1 substrate carries `research_only=true` until per-substrate symposium PROCEED + first paired smoke. No `dispatch_enabled: true` flip in Phase 3.
- **Substrate MUST be at OPTIMAL FORM before paid empirical dispatch**: Phase 1 cargo-cult audit IS the canonical OPTIMAL-FORM iteration per CLAUDE.md non-negotiable; Phase 3 implements the 4 FORK paths the audit identified.
- **PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium**: Phase 3 lands the BUILD that the per-substrate symposium will adjudicate; Phase 3 does NOT trigger dispatch.
- **MLX portable-local-substrate authority**: Phase 3 MLX-local CHEAP-SIGNAL-FIRST probe is `[macOS-MLX research-signal]` per CLAUDE.md non-negotiable; all artifacts carry `score_claim=false` + `promotion_eligible=false` + `ready_for_exact_eval_dispatch=false`.
- **Strict-scorer-rule**: inflate.py preserves ZERO scorer imports per HNeRV parity discipline L9 (Phase 3 adds compress-side MLX SegNet probe at MLX_iteration.py extension, NOT at inflate.py).
- **Public Disclosure Hygiene**: no /Users/adpena paths in any Phase 3 artifact.
- **Forbidden premature KILL without research exhaustion**: Phase 1 audit DEFERS the 4 critical cargo-cults to per-substrate symposium UNWIND-TEST scheduling; Phase 3 implements the unwind paths so the empirical anchor can adjudicate them.

---

## Sister artifacts preserved (no mutation per Catalog #110/#113 APPEND-ONLY)

Phase 3 will NOT mutate:
- `src/tac/substrates/nscs06_v8_chroma_lut/architecture.py` (sister-landed; HARD-EARNED architecture)
- `src/tac/substrates/nscs06_v8_chroma_lut/archive.py` (sister-landed; HARD-EARNED archive grammar)
- `src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py` (sister-landed; PRESERVE; Phase 3 ADDS sister `gt_distribution_matched_seed.py` alongside)
- `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py` (sister-landed; PRESERVE per-assumption ablation ladder)
- `src/tac/substrates/nscs06_v8_chroma_lut/substrate_contract.py` (sister-landed; PRESERVE)
- `src/tac/substrates/nscs06_v8_chroma_lut/__init__.py` (sister-landed; Phase 3 EXTENDS via NEW symbols imported alongside existing)
- `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (predecessor-landed; PRESERVE; Phase 3 EXTENDS the parity-verifier API surface)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_substrate.py` (sister-landed; PRESERVE 49 tests)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_revisions.py` (sister-landed; PRESERVE)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_mlx_iteration.py` (predecessor-landed; PRESERVE 31 tests including 3 real MLX integration tests)
- `experiments/train_substrate_nscs06_v8_chroma_lut.py` (sister-landed; Phase 3 does NOT modify trainer per scope; cls_stream wire-in to L0 inflate may require trainer update — DEFERRED to operator)
- `.omx/research/nscs06_v8_chroma_lut_design_20260521.md` (sister-landed; PRESERVE per HISTORICAL_PROVENANCE)

Phase 3 will ADD:
- `src/tac/substrates/nscs06_v8_chroma_lut/distinguishing_feature_smoke.py` (NEW; ~150 LOC)
- `src/tac/substrates/nscs06_v8_chroma_lut/gt_distribution_matched_seed.py` (NEW; ~120 LOC)
- `src/tac/substrates/nscs06_v8_chroma_lut/predicted_band_axis_attribution.py` (NEW; ~80 LOC)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_path_3_c_prime_cargo_cult_unwinds.py` (NEW; ~250 LOC; 10+ tests)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md` (NEW; Phase 3 design memo per 8-section discipline)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` (NEW; Phase 3 landing memo per Catalog #229 PV)

Phase 3 will EXTEND (canonical APPEND-ONLY sister discipline):
- `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (ADD `measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline` + sister verdict dataclass; NO mutation of existing API)
- `src/tac/substrates/nscs06_v8_chroma_lut/__init__.py` (EXTEND `__all__` with NEW symbol exports; NO mutation of existing exports)

---

## Cost + wall-clock estimate

| Phase | Cost | Wall-clock |
|---|---|---|
| Phase 1 (LANDED commit `a6e2a06e3`) | $0 | ~30 min |
| Phase 2 (this memo) | $0 | ~15 min |
| Phase 3 (L0/L1 SCAFFOLD + tests + design memo + landing memo) | $0 | ~2-3h |
| **TOTAL** | **$0** | **~3-4h** |

All paid empirical work (Modal dispatch + paired-CUDA-CPU eval) DEFERRED to per-substrate symposium PROCEED + operator-routable per CLAUDE.md "Executing actions with care".

---

## CLAUDE.md compliance

- Catalog #229 PV: read Phase 1 audit before Phase 2 decision
- Catalog #290 canonical-vs-unique decision per layer: 4 FORK rationales tabled
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: Phase 3 adds new files; preserves all sister + predecessor work
- Catalog #303 cargo-cult audit per assumption: cites Phase 1 audit; 4 unwind paths designed
- Catalog #297 signal-axis-destruction reversibility: 3a distinguishing-feature smoke IS the canonical reversibility-test surface for v8
- Catalog #324 predicted-band post-training validation: 3d axis-labeling discipline addresses cargo-cult #12
- Catalog #287 + #323 canonical Provenance: NO score claim asserted; all references tagged `[prediction]` or `[research-signal]`
- Catalog #325 per-substrate symposium: Phase 3 BUILD enables symposium adjudication; does NOT trigger dispatch
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM": Phase 1+2+3 IS the OPTIMAL FORM iteration cycle per directive #2
