<!-- SPDX-License-Identifier: MIT -->
---
schema_version: substrate_recursive_adversarial_review_memo_v2_20260516
deliberation_id: path_3_i_recursive_adversarial_review_r1_prime_prime_3_axis_20260526
substrate_id: faiss_ivf_pq_residual
review_round: R1''
per_substrate_counter_before: 0/3
per_substrate_counter_after: 0/3
verdict: NOT_CLEAN_FIX_WAVE_REQUIRED
landing_under_review_commits: [a883a717c, 587e3b85a, c4d8bbae8]
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Jegou
  - Schmid
  - Mallat
  - Atick
  - Contrarian
  - AssumptionAdversary
  - PR95Author
  - Hotz
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_assumption_adversary_verdict:
  - assumption: "Substrate package is MLX-first per landing memo §5 Axis 2 'MLX-first parity gate at Catalog #1265 threshold 0.001'"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Inspection of src/tac/substrates/faiss_ivf_pq_residual/mlx_renderer.py reveals NO MLX primitives are actually used in the substrate package; only `_ensure_mlx_available()` import guard + cost estimators + a `_full_main` that raises NotImplementedError. The 'MLX↔numpy parity test PASSES byte-identical' claim in landing memo §4 refers to a test that uses `mx.take` DIRECTLY in `tests/test_basic.py:318-326` — NOT through any substrate-package MLX module. The substrate is currently numpy-only at the package level; there is nothing to gate per Catalog #1265 MLX-first contest-equivalence."
  - assumption: "V1-V8 prior work is research INPUT only and the NEW substrate-design REDIRECT is principled per Phase 2"
    classification: HARD-EARNED-AXIOMATICALLY-VERIFIED
    rationale: "Phase 1 audit Layer 1 META-CC verdict is sound: V1-V8 targeted DIFFERENT substrate surface (side-info channel) than per-pair RGB residual codec stacking on PR110 fec6. Phase 2 Path (b) SUBSTRATE-DESIGN REDIRECT is the canonical structural fix. The redirect honors UNIQUE-AND-COMPLETE-PER-METHOD operating mode (CLAUDE.md non-negotiable) and rejects the path-of-least-resistance V1 extension."
  - assumption: "FAISSPQ1 archive grammar is byte-deterministic per Catalog #139/#220/#272"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Test suite confirms: test_header_size_constant + test_roundtrip + test_deterministic_bytes + test_magic_mismatch_raises + test_truncated_raises + test_meta_contains_canonical_keys all PASS. test_archive_byte_mutation_codebook_changes_decoded_reconstruction confirms operational consumption per Catalog #139 byte-mutation discipline. 29-byte header + 3 brotli-compressed blobs + meta is reproducible across runs."
  - assumption: "Jégou-Douze-Schmid 2011 PQ paradigm anchor is applicable to per-pair RGB residual"
    classification: HARD-EARNED-BY-PRINCIPLE-PENDING-EMPIRICAL
    rationale: "PQ paradigm is mathematically sound for vector quantization of high-dimensional inputs; per-pair RGB residual at tile-level (e.g. 4x4x3=48 dimensions per tile per design memo) is a valid PQ input. HOWEVER, the predicted byte budget envelope (recalibrated to ~17 tiles/pair feasible at ≤30 byte/pair per landing memo §2) lacks empirical anchor — Phase 4 PQ codebook training probe (op-routable #3) is required to verify the predicted band [0.180, 0.210] is structurally achievable."
council_decisions_recorded:
  - "OP-1: FIX-WAVE-R1''-I MUST add ACTUAL MLX primitives to mlx_renderer.py (currently only contains config dataclass + import guard); landing memo Axis 2 claim is empty without MLX primitives"
  - "OP-2: After OP-1, add MLX↔numpy parity test for substrate-package primitives (currently test_mlx_numpy_parity_pq_codebook_gather uses mx.take directly in test file, not through substrate API)"
  - "OP-3: WAVE-1 canonical posterior emission wire-in is MISSING for I; sister Wave-2 follow-on needed"
  - "OP-4: Phase 4 PQ codebook training probe + MLX-first parity gate at Catalog #1265 threshold 0.001 (per landing memo §7 step 4) is the canonical disambiguator"
council_predicted_mission_contribution: frontier_breaking_enabler
horizon_class: frontier_pursuit
deferred_substrate_id: faiss_ivf_pq_residual
related_deliberation_ids:
  - path_3_i_v1_faiss_ivf_pq_L0_scaffold_landed_20260526
  - path_3_i_v1_faiss_ivf_pq_cargo_cult_audit_20260526
  - path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
audit_evidence_tag: "[macOS-MLX research-signal]"
---

# Path 3 I — V1 Faiss IVF-PQ residual codec — R1'' recursive adversarial review (3-axis)

**Lane:** `lane_path_3_recursive_adversarial_review_r1_prime_prime_3_axis_landings_h_i_j_k_20260526` L1
**Predecessor counter:** 0/3 (NEW substrate; R1'' is first recursive review)
**Successor counter target:** 1/3 if CLEAN; reset 0/3 if NOT CLEAN
**This round verdict:** NOT_CLEAN_FIX_WAVE_REQUIRED → per-substrate counter REMAINS 0/3

## 1. Scope

R1'' review of the I=V1 Faiss IVF-PQ residual codec substrate landing per the operator binding 3-axis discipline.

## 2. Landings under review

| Commit | Description |
|---|---|
| `a883a717c` | Phase 1 cargo-cult audit (META: V1 surface different than per-pair RGB residual) |
| `587e3b85a` | Phase 2 substrate-design decision (Path b SUBSTRATE-DESIGN REDIRECT) |
| `c4d8bbae8` | Phase 3 L0 SCAFFOLD landed (8 files + smoke trainer + 20/20 tests) |

Substrate package: `src/tac/substrates/faiss_ivf_pq_residual/` (5 files in package + tests, total ~1520 LOC per landing memo §3).

## 3. Axis 1 — Math + scientific + engineering rigor per layer

### 3.1 HARD-EARNED layers (verified)

| Layer | Anchor | Verdict |
|---|---|---|
| Phase 2 SUBSTRATE-DESIGN REDIRECT (Path b FORK) | V1-V8 targeted side-info channel; I targets per-pair RGB residual codec stacking on PR110 fec6 | HARD-EARNED-AXIOMATICALLY-VERIFIED |
| Jégou-Douze-Schmid 2011 PQ as paradigm anchor | Vector quantization is canonical for high-dimensional input compression | HARD-EARNED-CANONICAL |
| FAISSPQ1 archive grammar byte-deterministic | 6 archive tests PASS + 1 byte-mutation test PASS | HARD-EARNED-EMPIRICALLY-VERIFIED |
| Mallat wavelet residual + Atick-Redlich retinal residual MI | Per-pair RGB residual signal characterization | HARD-EARNED-BY-PRINCIPLE |
| Catalog #146 inflate runtime contract | `inflate.py` honors 3-positional-arg `inflate.sh <archive_dir> <output_dir> <file_list>` | HARD-EARNED-EMPIRICALLY-VERIFIED |
| Catalog #205 canonical `select_inflate_device` | Adopted via canonical helper import | HARD-EARNED-CANONICAL |
| Catalog #240(c) `_full_main raises NotImplementedError` | Verified empirically per test_basic.py | HARD-EARNED-EMPIRICALLY-VERIFIED |

### 3.2 CARGO-CULTED-PENDING-EMPIRICAL

| Layer | Cargo-cult risk | Unwind path |
|---|---|---|
| Predicted byte budget [0.180, 0.210] frontier-pursuit (RECALIBRATED §9) | Lacks empirical anchor; Phase 4 PQ codebook training probe required | Phase 4 op-routable #3 in landing memo §7 |
| ~17 tiles/pair feasible at ≤30 byte/pair | Closed-form estimate; lacks empirical PQ codebook fit distortion measurement | Phase 4 PQ codebook training probe |
| Per-pair RGB residual is the correct PQ input signal | Hypothesis principled but not empirically validated against PR110 frontier residual extraction | Phase 4 op-routable #2 in landing memo §7 |

### 3.3 Findings

**NONE on Axis 1.** Math + scientific framing is sound. The SUBSTRATE-DESIGN REDIRECT is the canonical structural fix for the META cargo-cult of V1 surface vs per-pair RGB residual surface.

## 4. Axis 2 — MLX drift minimization per primitive

### 4.1 CRITICAL FINDING I-R1''-1 — MLX renderer is empty of MLX primitives

**Location:** `src/tac/substrates/faiss_ivf_pq_residual/mlx_renderer.py:1-219` (entire file)

Inspection reveals:

| Section | Lines | Content |
|---|---|---|
| Module docstring | 1-19 | Describes MLX-first per design memo, cites sister A=DreamerV3 forensic |
| Imports | (top) | `from typing import Any` + lazy MLX import via `_ensure_mlx_available()` |
| `FaissIVFPQResidualConfig` dataclass | 43-141 | Config + cost estimators (NO MLX usage) |
| `_ensure_mlx_available()` | 144-156 | Lazy import guard (returns `mx` module object IF MLX installed) |
| `estimate_per_pair_codeword_bytes_raw` | 159-164 | Pure-Python integer arithmetic (NO MLX usage) |
| `estimate_archive_bytes` | 167-190 | Pure-Python integer arithmetic + float multiplication (NO MLX usage) |
| `_full_main` | 193-209 | Raises NotImplementedError per Catalog #240 |
| `__all__` | 212-219 | Public API listing |

**Bug class:** Per landing memo §5 Axis 2: *"MLX-Faiss adapter feasibility | Integer codebook lookup deterministic; float gather deterministic | Sister G=NIRVANA mlx_renderer.py canonical pattern | sister G landed `f7d2e86fe` provides pattern; MLX↔numpy parity test PASSES"*. However the substrate package's `mlx_renderer.py` contains NO MLX primitives. The "MLX↔numpy parity test" in `tests/test_basic.py:318-326` uses `mx.take` directly via the test's own imports — bypassing the substrate package entirely.

**Consequence:** The Catalog #1265 MLX-first contest-equivalence gate (per op-routable #1 in landing memo §7 step 4) has NOTHING TO GATE. The substrate is numpy-only at the package level. The "MLX-first" framing is structurally vacuous.

**Cargo-cult classification:** CARGO-CULTED-EMPIRICALLY-FALSIFIED — landing memo §5 Axis 2 claims do not match the empirical substrate-package state.

**Fix path (FIX-WAVE-R1''-I):**
1. Implement actual MLX primitives in `mlx_renderer.py` for the substrate-relevant operations: `mlx_pq_codebook_gather`, `mlx_tile_reassemble`, `mlx_pq_encode` per the sister G=NIRVANA pattern
2. Add MLX↔numpy parity test in `tests/test_basic.py` covering the substrate-package primitives (not just `mx.take` directly in test file)
3. Empirically verify drift bound per Catalog #1265 threshold 0.001 at the substrate-package surface
4. Update landing memo §5 Axis 2 to reflect actual MLX usage

### 4.2 Test-file MLX usage (out of substrate scope)

The test file at `src/tac/substrates/faiss_ivf_pq_residual/tests/test_basic.py:318-326` uses `mx.take` directly:

```python
codebook_mx = mx.array(codebook_np)
...
sub_gathered = mx.take(codebook_mx[m], m_indices_mx, axis=0)  # (5, sub_dim)
gathered_mx = mx.stack(gathered_mx_per_sub, axis=1)
```

This is a test-level MLX comparison, not substrate-package MLX usage. The substrate package itself remains MLX-free.

### 4.3 Note on Faiss-vs-MLX distinction (per brief)

The brief raised the question: "I=V1 Faiss IVF-PQ residual: per-MLX-primitive (note: Faiss is C++ external; what's MLX vs numpy?)". The empirical answer:

- **Faiss is OPTIONAL accelerator** for PQ codebook training (`train_pq_codebook` in `numpy_reference.py` is numpy-only K-means; Faiss-CPU can swap in for performance per V4 hand-rolled probe OMP_NUM_THREADS=1 workaround per Phase 1 audit assumption #7)
- **MLX is the supposed-INFLATE-TIME-ACCELERATOR** for PQ codebook gather + tile reassemble per the design memo
- **numpy is the AXIS-3 PORTABLE REFERENCE** — and is currently the ONLY implementation in the substrate package

The substrate's *paradigm* (Jégou-Douze-Schmid PQ) is C++-Faiss-paper-canonical; the substrate's *implementation* is numpy-only. There is no MLX primitive code in `mlx_renderer.py` to verify.

## 5. Axis 3 — Portability via numpy per primitive

### 5.1 Per-primitive numpy reference verification

| Primitive | numpy reference | Status |
|---|---|---|
| `pq_codebook_gather` | `numpy_reference.py:pq_codebook_gather` | ✓ Present + tested (test_pq_codebook_gather_roundtrip PASSES) |
| `tiles_to_frame_nhwc` / `frame_to_tiles_nhwc` | `numpy_reference.py:tiles_to_frame_nhwc/frame_to_tiles_nhwc` | ✓ Present + tested |
| `train_pq_codebook` (numpy K-means) | `numpy_reference.py:train_pq_codebook` | ✓ Present (Faiss-CPU optional accelerator) |
| `encode_per_pair_residual` | `numpy_reference.py:encode_per_pair_residual` | ✓ Present + tested |
| `pq_reconstruct_tile_vectors` | `numpy_reference.py:pq_reconstruct_tile_vectors` | ✓ Present + tested |
| `to_uint8` | `numpy_reference.py:to_uint8` | ✓ Present + tested |
| Bilinear upsample (inflate-time) | INFLATE-RUNTIME via PyTorch per Catalog #146 | ✓ Catalog #146 canonical |

### 5.2 Portability evidence

- `numpy_reference.py` is 11.2 KB / ~280 LOC; complete substrate-package primitive set per landing memo §3
- Substrate operable on CPU-only test rigs without MLX OR Faiss dependency per landing memo §5
- GHA CPU CI testing supported via numpy-only path

### 5.3 Findings

**NONE on Axis 3.** Portability discipline is exemplary — the numpy reference is COMPLETE and tested. The portability claim is structurally correct.

## 6. Cross-substrate META findings (review-context only)

| META finding | Surface | Notes |
|---|---|---|
| WAVE-1 posterior emission wire-in MISSING for I | I `__init__.py` has 0 references to `emit_landing_posterior_anchor` / `posterior_emission_helper` | I landed at 03:16 BEFORE WAVE-1 wire-in at 04:01; WAVE-1 covered A/B'/C'/D/E/F/G/H (8 substrates) per commit `3d103dafd` message but did NOT include I (or J or K). Sister Wave-2 follow-on op-routable in aggregate memo. |
| Substrate-package vs test-file MLX usage divergence | `mlx_renderer.py` vs `tests/test_basic.py:318-326` | Empirical Axis 2 claims in landing memo §5 are structurally based on TEST-FILE MLX usage, not SUBSTRATE-PACKAGE MLX usage. This is the canonical observability surface gap. |
| Catalog #240(c) posture verified | `_full_main` raises NotImplementedError | CORRECT POSTURE per substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY |

## 7. R1'' verdict + per-substrate counter

### Verdict: NOT_CLEAN_FIX_WAVE_REQUIRED

**Reason:** 1 CRITICAL Axis 2 finding (I-R1''-1: MLX renderer contains NO actual MLX primitives; Axis 2 claims in landing memo §5 are structurally vacuous). The substrate is currently NOT MLX-first at the package level despite design memo + landing memo claims to be so.

### Per-substrate counter

- Before R1'': 0/3
- After R1'': **0/3 (RESET due to 1 CRITICAL finding)**
- Path to 1/3: FIX-WAVE-R1''-I lands actual MLX primitives in substrate package + verified empirical parity test

### Successor required

**FIX-WAVE-R1''-I** subagent must:
1. Implement substrate-package MLX primitives (mlx_pq_codebook_gather, mlx_tile_reassemble, mlx_pq_encode) per sister G=NIRVANA pattern
2. Add MLX↔numpy parity test in `tests/test_basic.py` covering substrate-package primitives (not just `mx.take` in test file)
3. Empirically verify drift bound per Catalog #1265 threshold 0.001 at the substrate-package surface
4. Update landing memo §5 Axis 2 to reflect actual MLX-package usage

## 8. 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = ACTIVE (per-tile PQ codeword distribution as per-tile sensitivity primitive once substrate-package MLX primitives land)
- hook #2 Pareto constraint = N/A (defensive review memo)
- hook #3 bit-allocator = N/A
- hook #4 cathedral autopilot dispatch = ACTIVE (substrate-package vs test-file MLX divergence informs autopilot's Catalog #341 routing-marker weight)
- hook #5 continual-learning posterior = ACTIVE (this memo's frontmatter consumable by `tac.council_continual_learning.append_council_anchor`)
- hook #6 probe-disambiguator = ACTIVE (Catalog #1265 gate IS the disambiguator BUT requires actual MLX primitives in substrate package to operate)

## 9. Discipline compliance

- ✅ Catalog #229 PV (read landing memo + 3 substrate source files + tests BEFORE writing memo)
- ✅ Catalog #110/#113 APPEND-ONLY (NEW review memo only; sister landing memos NEVER mutated)
- ✅ Catalog #208 docs/local-paths (no `/Users/` absolute paths)
- ✅ Catalog #230 sister-subagent ownership map (review-only; no file modifications per brief)
- ✅ Catalog #287 placeholder-rationale rejection (every assumption_adversary_verdict carries substantive ≥4-char rationale)
- ✅ Catalog #292 per-axis assumption surfacing (4 assumptions classified)
- ✅ Catalog #300 v2 frontmatter complete (tier T2; 10 attendees; quorum met)
- ✅ Catalog #340 sister-checkpoint guard PROCEED (review-only)
- ✅ Per CLAUDE.md "Executing actions with care": review-only NO code modifications

## 10. Cross-references

- Landing memo: `.omx/research/path_3_i_v1_faiss_ivf_pq_L0_scaffold_landed_20260526.md`
- Phase 1 audit: `.omx/research/path_3_i_v1_faiss_ivf_pq_cargo_cult_audit_20260526.md`
- Phase 2 decision: `.omx/research/path_3_i_v1_faiss_ivf_pq_substrate_design_decision_20260526.md`
- Sister G=NIRVANA scaffold pattern: `src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py`
- WAVE-1 posterior emission helper: commit `f6b432be1`; wire-in commit `3d103dafd`
- Path 3 R1' aggregate (sister round): `.omx/research/path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526.md`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
