<!-- SPDX-License-Identifier: MIT -->
---
schema_version: substrate_recursive_adversarial_review_memo_v2_20260516
deliberation_id: path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526
substrate_id: coin_pp_implicit_neural_representation
review_round: R1''
per_substrate_counter_before: 0/3
per_substrate_counter_after: 0/3
verdict: NOT_CLEAN_FIX_WAVE_REQUIRED
landing_under_review_commits: [eadee66ae]
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - PR95Author
  - Time-Traveler
  - Carmack
  - Hotz
  - Contrarian
  - AssumptionAdversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_assumption_adversary_verdict:
  - assumption: "K=COIN++ landing-memo empirical MLX matmul drift anchor = 5e-3 is structurally achievable on M-series MPS hardware"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Landing memo §3 Axis 2 line 116: 'Empirical anchor LANDED: MLX matmul drift = 5e-3 on M-series MPS (vs predicted 1e-5)'. Independent verification on this machine at typical K substrate dimensions (32x32, 64x64, 128x128, 256x64, 64x256 matmuls) returned drift bounds: 1.42e-2 (32x32) / 2.44e-2 (64x64) / 3.95e-2 (128x128) / 2.82e-2 (256x64) / 5.19e-2 (64x256). All measurements are O(1e-2) abs — ONE ORDER OF MAGNITUDE WORSE than the claimed 5e-3 bound. Relative drift is consistent at ~7-9e-4 which IS bounded, but the absolute claim is empirically false. The test_mlx_numpy_parity_skipped_if_mlx_unavailable test (per landing memo) asserts ≤ 5e-3; this assertion will likely FAIL on tested hardware classes at the default substrate dimensions if run empirically."
  - assumption: "K=COIN++ MLX renderer AVOIDS canonical anti-patterns (mx.repeat upsample, align_corners=True, mx.softmax-without-eps, non-Kahan large-N, fp16 matmul)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "grep src/tac/substrates/coin_pp_implicit_neural_representation/mlx_renderer.py: ZERO mx.repeat references (only in docstring negation: 'NO mx.repeat 2× upsample (not used; substrate is coordinate-batched, not grid-upsampled)'); ZERO align_corners=True; ZERO mx.softmax; ZERO mx.float16. Substrate is coordinate-batched per coord_mlp_forward → no upsample needed. The 'AVOIDED anti-patterns' enumeration in landing memo §3 Axis 2 line 110-114 is empirically verified."
  - assumption: "Coord-MLP + FiLM + sinusoidal positional encoding architectural choices are HARD-EARNED across math+scientific+engineering axes per design memo"
    classification: HARD-EARNED-CANONICAL
    rationale: "COIN++ (Dupont et al. 2022) is the canonical meta-learned modulated INR for image compression. Sinusoidal positional encoding (NeRF/COIN canonical), FiLM modulation (Perez 2017), sigmoid output activation — all are field-canonical primitives with first-principles + paper-citation provenance. The 10/10 HARD-EARNED claim in landing memo §3 Axis 1 holds for the architectural primitives. CARGO-CULTED carve-out for MOD_DIM=64 specific choice is properly disclosed."
  - assumption: "Numpy reference is COMPLETE at substrate-package level and supports CPU-only GHA CI"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Inspection of src/tac/substrates/coin_pp_implicit_neural_representation/numpy_reference.py confirms 9 canonical primitives: to_float32, linear, sin, cos, sinusoidal_positional_encoding, film_modulate, sigmoid, make_coord_grid_nhwc, mean/kahan_mean, plus composite coord_mlp_forward. Pure numpy with no MLX/torch import per Axis 3. Tests 26/26 PASS confirms numpy-only path operable on this machine (which has MLX installed; the numpy reference would also pass on a CPU-only rig per landing memo claim)."
council_decisions_recorded:
  - "OP-1: FIX-WAVE-R1''-K MUST update landing memo §3 Axis 2 line 116 empirical anchor from 5e-3 to empirically verified range (e.g. 1e-2 to 5e-2 abs / 1e-3 rel) per independent verification; update test_mlx_numpy_parity_skipped_if_mlx_unavailable assertion threshold to match empirical reality"
  - "OP-2: WAVE-1 canonical posterior emission wire-in MISSING for K; sister Wave-2 follow-on needed"
  - "OP-3: At L1 PROMOTION, run MOD_DIM sweep per landing memo §Operator-routable #3 to verify MOD_DIM=64 sufficiency (currently CARGO-CULTED per design memo Catalog #303 #2)"
council_predicted_mission_contribution: frontier_breaking_enabler
horizon_class: frontier_pursuit
deferred_substrate_id: coin_pp_implicit_neural_representation
related_deliberation_ids:
  - path_3_k_coin_pp_L0_scaffold_landed_20260526
  - path_3_k_coin_pp_substrate_design_20260526
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
audit_evidence_tag: "[macOS-MLX research-signal]"
---

# Path 3 K — COIN++ implicit neural representation — R1'' recursive adversarial review (3-axis)

**Lane:** `lane_path_3_recursive_adversarial_review_r1_prime_prime_3_axis_landings_h_i_j_k_20260526` L1
**Predecessor counter:** 0/3 (NEW substrate; R1'' is first recursive review)
**Successor counter target:** 1/3 if CLEAN; reset 0/3 if NOT CLEAN
**This round verdict:** NOT_CLEAN_FIX_WAVE_REQUIRED → per-substrate counter REMAINS 0/3

## 1. Scope

R1'' review of the K=COIN++ implicit neural representation substrate landing per the operator binding 3-axis discipline.

## 2. Landings under review

| Commit | Description |
|---|---|
| `eadee66ae` | Path 3 K COIN++ implicit neural representation L0 SCAFFOLD MLX-first fresh design |

Substrate package: `src/tac/substrates/coin_pp_implicit_neural_representation/` (5 files in package + tests + smoke trainer, total ~1500 LOC per landing memo).

## 3. Axis 1 — Math + scientific + engineering rigor per layer

### 3.1 HARD-EARNED layers (verified)

| Layer | Anchor | Verdict |
|---|---|---|
| Coord-MLP architecture (COIN++) | Dupont et al. 2022 canonical meta-learned modulated INR | HARD-EARNED-CANONICAL |
| FiLM modulation | Perez 2017 canonical | HARD-EARNED-CANONICAL |
| Sinusoidal positional encoding | NeRF/COIN canonical | HARD-EARNED-CANONICAL |
| Sigmoid output activation | Numerically stable per `numpy_reference.py:sigmoid` | HARD-EARNED-CANONICAL |
| brotli q=9 archive compression | Sister-canonical (mirrors NIRVANA1 + DREAMER1) | HARD-EARNED-CANONICAL |
| fp16 state_dict serialization | Sister-canonical pattern | HARD-EARNED-CANONICAL |
| Catalog #205 canonical `select_inflate_device` | Adopted via canonical helper import | HARD-EARNED-CANONICAL |
| Catalog #164 score_pair_components | Canonical scorer-loss routing | HARD-EARNED-CANONICAL |
| Kahan summation helper | Numerically stable for large-N reductions | HARD-EARNED-NUMERICAL |
| Catalog #240(c) `_full_main raises NotImplementedError` | Verified empirically per test_basic.py | HARD-EARNED-EMPIRICALLY-VERIFIED |

### 3.2 CARGO-CULTED-PENDING-EMPIRICAL

| Layer | Cargo-cult risk | Unwind path |
|---|---|---|
| MOD_DIM=64 specific value | Properly disclosed as CARGO-CULTED in landing memo §3 Axis 1 line 102; CARGO-CULTED per Catalog #303 audit | Phase 2+ MOD_DIM sweep per landing memo §Operator-routable #3 |
| Empirical MLX matmul drift 5e-3 claim | Independent verification shows 1e-2 to 5e-2 absolute (10× worse) | FIX-WAVE-R1''-K op-routable below |
| MOD_DIM×8 bits per pair budget allocation | Bit-allocator surface trivial per landing memo §6 hook #3 | Phase 2+ empirical verification |

### 3.3 Findings

**NONE on Axis 1.** Architectural primitives are HARD-EARNED canonical. The 8 ADOPT_CANONICAL + 3 FORK_BECAUSE_PRINCIPLED_MISMATCH per-layer decision matrix per landing memo §"Catalog #290 per-layer canonical-vs-unique decisions summary" is exemplary.

## 4. Axis 2 — MLX drift minimization per primitive

### 4.1 Per-primitive drift verification

| Primitive | MLX path | Empirical drift | Anti-pattern? |
|---|---|---|---|
| `linear` (matmul) | `mx.matmul` (default fp32) | **1.42e-2 (32x32) to 5.19e-2 (64x256) on this M-series machine — see §4.2** | NO (canonical, but bound understated) |
| `sin`/`cos` (positional encoding) | `mx.sin` / `mx.cos` (explicit fp32) | bit-exact (sin drift ~1e-7 verified) | NO |
| `sinusoidal_positional_encoding` (composite) | composite per-primitive | bounded by linear + sin/cos | NO |
| `film_modulate` | elementwise multiply + add | bit-exact-equivalent | NO |
| `sigmoid` (output) | `mx.sigmoid` (explicit fp32) | bit-exact-equivalent | NO |
| `make_coord_grid_nhwc` | `mx.linspace` + broadcast | bit-exact | NO |
| `mean` / `kahan_mean` | Kahan summation helper available | bounded by O(1) ULP | NO |
| `coord_mlp_forward` (composite) | composite per-primitive | bounded by linear + sin/cos + film + sigmoid | NO |

### 4.2 CRITICAL FINDING K-R1''-1 — 5e-3 matmul drift anchor empirically falsified

**Location:** Landing memo §3 Axis 2 line 116 + test assertion in `tests/test_basic.py`

**Claim:** *"Empirical anchor LANDED: MLX matmul drift = 5e-3 on M-series MPS (vs predicted 1e-5 in design memo). This documents REAL HARDWARE BEHAVIOR; sister A=DreamerV3 max_abs=24.34 was 4 orders of magnitude worse due to align_corners=True bilinear + mx.repeat anti-pattern. The 5e-3 bound is the residual hardware-induced bound after AVOIDING anti-patterns."*

**Independent verification on this machine** at K-typical substrate dimensions:

```
matmul (64, 64)   drift abs=2.441597e-02 rel=7.990579e-04
matmul (32, 32)   drift abs=1.420593e-02 rel=6.405374e-04
matmul (128, 128) drift abs=3.950500e-02 rel=9.186353e-04
matmul (256, 64)  drift abs=2.821922e-02 rel=8.067542e-04
matmul (64, 256)  drift abs=5.193329e-02 rel=7.972131e-04
```

All absolute drifts are O(1e-2) — **ONE ORDER OF MAGNITUDE WORSE** than the claimed 5e-3 anchor. The 5e-3 claim is empirically false at typical substrate dimensions.

The RELATIVE drift (~7-9e-4) IS bounded and is consistent across substrate dimensions; this is the M-series MPS fp32 matmul hardware-class floor. The relative bound is canonical for the hardware.

**Bug class:** The landing memo's headline empirical anchor (line 116) is CARGO-CULTED-EMPIRICALLY-FALSIFIED on the same machine class used for the original measurement. This is likely an artifact of measuring a single small matmul (e.g. the test fixture's specific shape) and generalizing to "the bound is 5e-3" without per-dimension verification. The actual bound is dimension-dependent and runs 1e-2 to 5e-2 absolute at common substrate dimensions.

**Consequence:** The test_mlx_numpy_parity_skipped_if_mlx_unavailable assertion threshold (per landing memo line 116) is likely an under-estimate of the hardware drift, and the test could fail when run on different MLX matmul shapes than the test fixture.

**Cargo-cult classification:** CARGO-CULTED-EMPIRICALLY-FALSIFIED (claim does not survive independent verification).

**Fix path (FIX-WAVE-R1''-K):**
1. Re-measure MLX matmul drift across K-typical substrate dimensions (32x32, 64x64, 128x128, 256x64, 64x256, plus K's actual MOD_DIM=64 hidden=64 depth=3 chain)
2. Update landing memo §3 Axis 2 line 116 empirical anchor from 5e-3 to empirically verified range (e.g. "1e-2 to 5e-2 absolute / 7e-4 to 1e-3 relative" with per-dimension breakdown)
3. Update test assertion threshold to match empirical reality (use relative drift bound ~1e-3 instead of absolute 5e-3)
4. Add per-dimension drift characterization to substrate's `numpy_reference.py` docstring so future operators have accurate hardware-class expectations

### 4.3 Anti-pattern avoidance — VERIFIED

- ✗ `mx.repeat` upsample: **ZERO references** in mlx_renderer.py (substrate is coordinate-batched, not grid-upsampled)
- ✗ `align_corners=True` bilinear: **ZERO references**
- ✗ `mx.softmax` without epsilon: **ZERO references** (substrate is not softmax-based)
- ✗ Non-Kahan summation at large-N: Kahan helper available for queued use
- ✗ fp16 matmul without explicit fp32 accumulation: **ZERO references**

The AVOIDED anti-patterns enumeration in landing memo §3 Axis 2 line 110-114 IS empirically verified. The substrate is structurally clean of the anti-pattern bug class that bit sister A=DreamerV3 (pre-FIX-WAVE-R1).

### 4.4 Note on relative-vs-absolute drift framing

K's claim (5e-3 abs) is empirically false; K's TRUE bound (~7-9e-4 rel) is canonically correct for M-series MPS fp32 matmul. If the landing memo had framed the anchor as "relative drift ~1e-3", the claim would have been empirically clean. The fix is a framing correction, not an architectural concern.

## 5. Axis 3 — Portability via numpy per primitive

### 5.1 Per-primitive numpy reference verification

| Primitive | numpy reference | Status |
|---|---|---|
| `to_float32` | `numpy_reference.py:to_float32` | ✓ Present |
| `linear` | `numpy_reference.py:linear` | ✓ Present |
| `sin`/`cos` | `numpy_reference.py:sin/cos` | ✓ Present |
| `sinusoidal_positional_encoding` | `numpy_reference.py:sinusoidal_positional_encoding` | ✓ Present + tested |
| `film_modulate` | `numpy_reference.py:film_modulate` | ✓ Present |
| `sigmoid` | `numpy_reference.py:sigmoid` (numerically stable) | ✓ Present |
| `make_coord_grid_nhwc` | `numpy_reference.py:make_coord_grid_nhwc` | ✓ Present |
| `mean` / `kahan_mean` | `numpy_reference.py:mean/kahan_mean` | ✓ Present |
| `coord_mlp_forward` (composite) | `numpy_reference.py:coord_mlp_forward` | ✓ Present + tested |

### 5.2 Portability evidence

- `numpy_reference.py` is 12.7 KB / ~280 LOC; complete substrate-package primitive set
- 9/9 numpy reference implementations per landing memo §3 Axis 3 line 118-130
- Pure numpy with no MLX/torch import per Axis 3
- Tests 26/26 PASS (verified independently)

### 5.3 Findings

**NONE on Axis 3.** Portability discipline is exemplary. The Kahan summation helper is the canonical numerical-stability primitive for large-N reductions.

## 6. Cross-substrate META findings (review-context only)

| META finding | Surface | Notes |
|---|---|---|
| WAVE-1 posterior emission wire-in MISSING for K | K `__init__.py` has 0 references to `emit_landing_posterior_anchor` / `posterior_emission_helper` | K landed at 03:06 BEFORE WAVE-1 wire-in at 04:01; WAVE-1 covered A/B'/C'/D/E/F/G/H per commit `3d103dafd` but NOT K (despite K landing FIRST chronologically of K+I+J). Sister Wave-2 follow-on op-routable in aggregate memo. |
| Catalog #240(c) posture verified | `_full_main` raises NotImplementedError | CORRECT POSTURE |
| Sister `coin_plus_plus/` (2026-05-20 prior sketch) preserved | `src/tac/substrates/coin_plus_plus/` untouched | Per Catalog #110/#113 HISTORICAL_PROVENANCE; K substrate is fresh design at distinct path |

## 7. R1'' verdict + per-substrate counter

### Verdict: NOT_CLEAN_FIX_WAVE_REQUIRED

**Reason:** 1 CRITICAL Axis 2 finding (K-R1''-1: empirical 5e-3 matmul drift anchor falsified at typical substrate dimensions; actual drift is 1e-2 to 5e-2 absolute). This is a framing/measurement issue, not an architectural concern, but the landing memo's headline anchor must be corrected per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

### Per-substrate counter

- Before R1'': 0/3
- After R1'': **0/3 (RESET due to 1 CRITICAL finding)**
- Path to 1/3: FIX-WAVE-R1''-K lands empirical anchor correction + test assertion threshold update

### Successor required

**FIX-WAVE-R1''-K** subagent must:
1. Re-measure MLX matmul drift across K-typical substrate dimensions (32x32, 64x64, 128x128, 256x64, 64x256, plus K's actual MOD_DIM=64 hidden=64 depth=3 chain)
2. Update landing memo §3 Axis 2 line 116 empirical anchor with per-dimension breakdown (per CLAUDE.md "Apples-to-apples evidence discipline")
3. Update `tests/test_basic.py` assertion threshold to match empirical reality (use relative drift bound ~1e-3 instead of absolute 5e-3 OR pin to specific dimensions tested)
4. Add per-dimension drift characterization to `numpy_reference.py` docstring

## 8. 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = N/A at L0 SCAFFOLD (queued for L1+)
- hook #2 Pareto constraint = N/A at L0 (queued for L1+)
- hook #3 bit-allocator = N/A at L0 (per-pair modulation rate is fixed at MOD_DIM×8 bits)
- hook #4 cathedral autopilot dispatch = N/A at L0 (per Catalog #341 routing-markers all non-promotable)
- hook #5 continual-learning posterior = ACTIVE (frontmatter consumable by `tac.council_continual_learning.append_council_anchor`)
- hook #6 probe-disambiguator = ACTIVE (Catalog #1265 gate IS the disambiguator)

## 9. Discipline compliance

- ✅ Catalog #229 PV (read landing memo + 3 substrate source files + tests + ran empirical drift verification BEFORE writing memo)
- ✅ Catalog #110/#113 APPEND-ONLY (NEW review memo only; sister landing memos NEVER mutated)
- ✅ Catalog #208 docs/local-paths (no `/Users/` absolute paths)
- ✅ Catalog #230 sister-subagent ownership map (review-only)
- ✅ Catalog #287 placeholder-rationale rejection (every assumption_adversary_verdict carries substantive ≥4-char rationale)
- ✅ Catalog #292 per-axis assumption surfacing (4 assumptions classified)
- ✅ Catalog #300 v2 frontmatter complete (tier T2; 8 attendees; quorum met)
- ✅ Catalog #340 sister-checkpoint guard PROCEED
- ✅ Per CLAUDE.md "Executing actions with care": review-only NO code modifications
- ✅ Per CLAUDE.md "Apples-to-apples evidence discipline": empirical anchor framing must match empirical reality

## 10. Cross-references

- Landing memo: `.omx/research/path_3_k_coin_pp_L0_scaffold_landed_20260526.md`
- Phase 3 design memo: `.omx/research/path_3_k_coin_pp_substrate_design_20260526.md`
- Sister `coin_plus_plus/` (2026-05-20 prior sketch; HISTORICAL_PROVENANCE preserved): `src/tac/substrates/coin_plus_plus/`
- Sister G=NIRVANA canonical template: `src/tac/substrates/nirvana_cascading_nerv/`
- WAVE-1 posterior emission helper: commit `f6b432be1`; wire-in commit `3d103dafd`
- Path 3 R1' aggregate (sister round): `.omx/research/path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526.md`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
