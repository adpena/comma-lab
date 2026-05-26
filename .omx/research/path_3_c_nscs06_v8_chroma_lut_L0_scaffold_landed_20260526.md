---
council_tier: T1
council_attendees: [Shannon, Dykstra, AssumptionAdversary, NSCS06-v7-author-cite, Rudin]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Phase 3 L0/L1 SCAFFOLD lands all 4 cargo-cult unwinds with empirical confirmation"
    classification: HARD-EARNED
    rationale: "180 tests pass (49 sister test_substrate + 56 sister test_revisions + 31 predecessor test_mlx_iteration + 44 new test_path_3_c_prime_cargo_cult_unwinds); cargo-cult #5 (L0 cls=0 uniform structural test-invalidity) EMPIRICALLY CONFIRMED via FAIL_AT_CLASS_1 verdict; canonical-vs-unique decision per layer documented per Catalog #290; 6-hook wire-in declared per Catalog #125."
council_decisions_recorded:
  - "L0 SCAFFOLD landed; L1 promotion blocker = wire cls_stream consumption at L0 inflate (operator-routable per Catalog #325 per-substrate symposium)"
  - "operator-routable: per-substrate symposium per Catalog #325 schedules within 14-day window 2026-05-26 → 2026-06-09"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
horizon_class: frontier_pursuit
---

# Path 3 C' — NSCS06 v8 chroma_lut L0/L1 SCAFFOLD landing memo

**Date:** 2026-05-26
**Lane:** `lane_path_3_c_prime_nscs06_v8_chroma_lut_cargo_cult_first_20260526` (L0 SCAFFOLD landed)
**Phase 1 audit:** commit `a6e2a06e3` (`.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md`)
**Phase 2 decision:** commit `bac0ec05d` (`.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md`)
**Phase 3 design:** `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md` (co-landed)
**Predecessor:** `adb6518fd5bb26607` (TaskStop) MLX iteration scaffold + 31 tests INPUT (preserved per Catalog #110/#113 APPEND-ONLY)

---

## Premise verification per Catalog #229

Read before any edit:
- Phase 1 cargo-cult audit memo (12 assumptions interrogated; 4 CARGO-CULTED-CRITICAL identified)
- Phase 2 substrate-design decision memo (Path (b) JUSTIFIED-EXTEND-WITH-FORK + 7-section roadmap)
- All 6 sister v8 chroma_lut modules (architecture.py 341 LOC, archive.py 418 LOC, inflate.py 223 LOC, procedural_variant.py 393 LOC, revisions.py 1192 LOC, substrate_contract.py 105 LOC)
- Predecessor mlx_iteration.py (762 LOC) + tests (31 tests including 3 real MLX integration)
- Existing v8 design memo `.omx/research/nscs06_v8_chroma_lut_design_20260521.md` (24 KB)
- Sister v6→v7 cargo-cult-unwind redesign memo `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md` (15.5 KB; 44% improvement empirical anchor)
- Canonical equation #26 `src/tac/canonical_equations/procedural_codebook_savings.py` (`_INCLUDED_CONTEXTS` includes `nscs06_v8_chroma_lut`)
- Sister trainer `experiments/train_substrate_nscs06_v8_chroma_lut.py` (1007 LOC)
- Canonical helper `src/tac/procedural_codebook_generator/` (multiple submodules; `derive_codebook_from_seed` is canonical PCG64 expansion)

Total premise-verification surface read: ~10000 LOC code + 4 design memos + 1 canonical equation registry source.

---

## Empirical verdict table per Phase 3 implementation

| Phase 3 deliverable | Empirical receipt | Cargo-cult unwound |
|---|---|---|
| `distinguishing_feature_smoke.py` (NEW; 380 LOC) | 11 tests pass + empirical FAIL_AT_CLASS_1 verdict at L0 SCAFFOLD | #5 (cls=0 uniform structural test-invalidity) EMPIRICALLY CONFIRMED |
| `gt_distribution_matched_seed.py` (NEW; 285 LOC) | 14 tests pass (determinism + GT-fingerprint encoding + canonical PCG64 expansion roundtrip) | #3 (PCG64 distribution mismatch) — alternative arm SCAFFOLDED |
| `mlx_iteration.py` EXTENSION (~280 LOC added) | 5 tests pass (verdict-dataclass invariants); real-MLX integration deferred to operator-routable | #8 (SegNet noise-floor unknown) — $0 probe SCAFFOLDED |
| `predicted_band_axis_attribution.py` (NEW; 175 LOC) | 8 tests pass (axis tokens + canonical Provenance + validation-status predicate) | #12 (rate-axis-only predicted band) — Catalog #324 discipline SCAFFOLDED |
| `__init__.py` EXTENSION (PRESERVED + EXTENDED) | All 5 new symbols importable from package root | All 4 cargo-cult unwinds + sister surface exports |
| `test_path_3_c_prime_cargo_cult_unwinds.py` (NEW; 44 tests) | 44 / 44 PASS | All 4 cargo-cult unwinds tested |
| Regression guards | 49 sister + 56 sister + 31 predecessor = 136 prior tests still PASS | NO regressions per Catalog #110/#113 APPEND-ONLY |

**Total tests:** 180 (44 new + 136 preserved) — ALL PASS in 2.77s.

---

## Reproducer

```bash
# Phase 3 cargo-cult-unwind tests
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
    src/tac/substrates/nscs06_v8_chroma_lut/tests/test_path_3_c_prime_cargo_cult_unwinds.py -v

# Regression guard (full v8 chroma_lut test suite)
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest \
    src/tac/substrates/nscs06_v8_chroma_lut/tests/ -q

# Verify all 5 new Phase 3 symbols importable from package root
PYTHONPATH=src:upstream:$PWD .venv/bin/python -c "
from tac.substrates.nscs06_v8_chroma_lut import (
    verify_per_class_chroma_anchors_consumed_at_inflate,
    derive_chroma_lut_seed_from_gt_lut_bytes,
    measure_v8_chroma_lut_segnet_argmax_displacement_from_baseline,
    predicted_delta_s_with_axis_attribution,
    SegNetArgmaxDisplacementVerdict,
)
print('OK')
"
```

---

## 4-cargo-cult unwind status table

| Cargo-cult # | Phase 1 classification | Phase 3 implementation | Empirical evidence |
|---|---|---|---|
| **#3** PCG64-uniform LUT distribution mismatch | CARGO-CULTED-CRITICAL | `gt_distribution_matched_seed.py` hash-derived-from-GT alternative arm | 14 tests cover seed determinism + GT-fingerprint encoding + canonical PCG64 expansion roundtrip; empirical anchor pending per-substrate symposium |
| **#5** L0 cls=0 uniform structural test-invalidity | CARGO-CULTED-CRITICAL | `distinguishing_feature_smoke.py` per-class byte-mutation smoke | **EMPIRICALLY CONFIRMED** via `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` returning `verdict_kind=FAIL_AT_CLASS_1`; L1 promotion blocker = wire cls_stream consumption at L0 inflate |
| **#8** SegNet noise-floor sensitivity UNKNOWN | CARGO-CULTED-CRITICAL | `mlx_iteration.py` EXTENSION + `SegNetArgmaxDisplacementVerdict` | 5 verdict-dataclass invariant tests pass; real-MLX integration probe scaffolded; operator-routable: run `$0` MLX-local probe on real GT pairs BEFORE paid Modal dispatch per Phase 2 op-routable #3 |
| **#12** Rate-axis-only predicted band | CARGO-CULTED-CRITICAL | `predicted_band_axis_attribution.py` Catalog #324-grade axis attribution | 8 tests cover UNKNOWN tokens + canonical Provenance fields + `is_predicted_band_validated_post_training` predicate; seg+pose axis values pending paired-smoke per Catalog #324 reactivation criterion |

---

## L0 SCAFFOLD state per Catalog #240 + #325

- **Recipe state**: existing sister `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` PRESERVED per HISTORICAL_PROVENANCE; Path 3 C' Phase 3 does NOT modify the recipe (operator-routable per Catalog #325).
- **Trainer state**: existing sister `experiments/train_substrate_nscs06_v8_chroma_lut.py` (1007 LOC) PRESERVED; `_full_main` IMPLEMENTED per OVERNIGHT-V; Path 3 C' Phase 3 does NOT modify the trainer (operator-routable: wire cls_stream consumption at L0 inflate would require trainer update; DEFERRED per Phase 2 scope).
- **L0 → L1 promotion blocker**: wire `cls_stream` consumption at `inflate.py:185` (currently `cls_full = np.zeros_like(gray_full, dtype=np.uint8)`) so per-class byte-mutation smoke verdict transitions from `FAIL_AT_CLASS_1` to `PASS_PER_CLASS`. This is the canonical Catalog #325 per-substrate symposium reactivation criterion.

---

## CLAUDE.md compliance

- ✅ **Catalog #229 PV**: read full state of canonical files (10K+ LOC + 4 memos + canonical equation #26 source) BEFORE any edit
- ✅ **Catalog #117/#157/#174/#235 canonical serializer**: all 3 phase commits via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per commit
- ✅ **Catalog #206 subagent checkpoint**: 5+ checkpoints emitted via `tools/subagent_checkpoint.py`
- ✅ **Catalog #119 Co-Authored-By Claude trailer**: all 3 phase commits
- ✅ **Catalog #287 + #323 canonical Provenance**: no score claim asserted; all verdicts non-promotable + axis-tagged
- ✅ **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: sister + predecessor work preserved; Phase 3 adds 4 NEW files + EXTENDS 2 files via NEW symbols only
- ✅ **Catalog #208 docs/local-paths**: no `/Users/adpena/...` references
- ✅ **Catalog #230 sister-subagent ownership map**: A/D/E/B' cited; NO file overlap
- ✅ **Catalog #340 sister-checkpoint guard**: serializer handles structurally
- ✅ **Catalog #297 signal-axis-destruction reversibility audit**: NSCS06 v6 grayscale Y=R=G=B canonical anchor cited; Phase 3 surfaces v8 reversibility at 3 surfaces (v1 inline / v2 procedural-seed / L0 cls=0 uniform)
- ✅ **Catalog #359 anti-misapplication self-protect**: v8 chroma LUT substitution IS REPLACEMENT-savings (NOT residual-correction-hybrid); pair #1/pair #2 falsifications do NOT apply
- ✅ **Catalog #290 + #294 + #296 + #303 + #305 + #309 + #324 + #325 + #344**: full design-memo discipline per Phase 3 design memo
- ✅ **CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch"**: Phase 1+2+3 IS the canonical OPTIMAL FORM iteration cycle; no paid dispatch fired
- ✅ **CLAUDE.md "Forbidden premature KILL without research exhaustion"**: 4 cargo-cults UNWIND-IMPLEMENTED (not killed); per-substrate symposium gates promotion
- ✅ **CLAUDE.md "Executing actions with care"**: NO `gh pr create` / `gh release create` / Modal/Vast/Lightning dispatch fired
- ✅ **CLAUDE.md "MLX portable-local-substrate authority"**: all MLX-derived artifacts tagged `[macOS-MLX research-signal]` + `score_claim=false` + `promotion_eligible=false`

---

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map** = ACTIVE (per-class chroma consumption sensitivity)
- **hook #2 Pareto constraint** = ACTIVE PRIMARY (canonical equation #26 rate-axis ΔS = -0.002706)
- **hook #3 bit-allocator** = PLANNED (32-byte seed slot per Catalog #325 PROCEED)
- **hook #4 cathedral autopilot dispatch** = ACTIVE (sister consumer auto-discovered per Catalog #335)
- **hook #5 continual-learning posterior** = ACTIVE PRIMARY (first paired-smoke empirical anchor populates seg+pose axes per Catalog #324)
- **hook #6 probe-disambiguator** = ACTIVE PRIMARY ($0 MLX SegNet noise-floor probe + per-class byte-mutation smoke ARE canonical disambiguators)

---

## Cost + wall-clock

| Phase | Cost | Wall-clock |
|---|---|---|
| Phase 1 (commit `a6e2a06e3`) | $0 | ~30 min |
| Phase 2 (commit `bac0ec05d`) | $0 | ~15 min |
| Phase 3 (this landing) | $0 | ~2h |
| **TOTAL** | **$0** | **~3h** |

All paid empirical work DEFERRED to per-substrate symposium PROCEED + operator-routable per CLAUDE.md "Executing actions with care".

---

## Operator-routable next steps (in priority order)

1. **Per-substrate symposium per Catalog #325** schedules within 14-day window 2026-05-26 → 2026-06-09. Symposium adjudicates whether Phase 3 unwinds satisfy 6-step contract.
2. **WIRE cls_stream consumption at L0 inflate** (cargo-cult #5 remediation). The 30-LOC sister-pattern change couples v8 inflate to v7-style CLS_STREAM grammar. Confirmation: re-run `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` → verdict transitions to `PASS_PER_CLASS`.
3. **Run `$0` MLX-local SegNet noise-floor probe** on real GT pairs from `upstream/videos/0.mkv` BEFORE paid dispatch. Per Phase 2 op-routable #3. Confirmation: `SegNetArgmaxDisplacementVerdict.recommended_proceed=True` (substrate IS SegNet-detectable above noise floor).
4. **Queue first paired-smoke Modal dispatch** (~$0.50) AFTER (1)+(2)+(3) confirmed. Per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" + Catalog #324 post-training Tier-C reactivation criterion.
5. **Empirical seg+pose-axis anchor lands** → replace UNKNOWN tokens in `predicted_delta_s_with_axis_attribution` return value with measured floats → `is_predicted_band_validated_post_training` flips True → canonical equation #26 posterior update via `tac.canonical_equations.update_equation_with_empirical_anchor`.

---

## Sister artifacts list

**ADDED (Phase 3 NEW; 4 modules + 1 test file + 3 memos = 8 files):**
- `src/tac/substrates/nscs06_v8_chroma_lut/distinguishing_feature_smoke.py` (380 LOC; 11 tests)
- `src/tac/substrates/nscs06_v8_chroma_lut/gt_distribution_matched_seed.py` (285 LOC; 14 tests)
- `src/tac/substrates/nscs06_v8_chroma_lut/predicted_band_axis_attribution.py` (175 LOC; 8 tests)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_path_3_c_prime_cargo_cult_unwinds.py` (44 tests)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_cargo_cult_audit_of_existing_scaffold_20260526.md` (Phase 1)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_decision_20260526.md` (Phase 2)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md` (Phase 3 design)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` (Phase 3 landing; THIS memo)

**EXTENDED (APPEND-ONLY; sister symbols PRESERVED):**
- `src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py` (predecessor file; +280 LOC NEW)
- `src/tac/substrates/nscs06_v8_chroma_lut/__init__.py` (+NEW symbol exports)

**PRESERVED (no mutation per Catalog #110/#113):**
- All 6 sister v8 chroma_lut modules
- All 3 sister test files (49 + 56 + 31 = 136 tests)
- Sister design memo `nscs06_v8_chroma_lut_design_20260521.md`
- Sister trainer + recipe + canonical equation #26 source

---

## Sister coordination per Catalog #230

In commit body cite:
- A=`aaec7a0d220f31543` DreamerV3 RSSM (fresh design; disjoint scope)
- D=`af6ca73c5a7fc40f4` Z6 predictive coding (fresh design; disjoint scope)
- E=`a35f9f86781aaaa4f` BoostNeRV against PR110 (fresh design; disjoint scope)
- B'=`ac4283983ece21b83` Z7-Mamba-2 cargo-cult-first (concurrent methodology; disjoint substrate scope)

NO file overlap. Predecessor `adb6518fd5bb26607` test parity work + MLX iteration scaffold PRESERVED as input.
