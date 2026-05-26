<!-- SPDX-License-Identifier: MIT -->
---
schema_version: substrate_fix_wave_landing_memo_v2_20260516
deliberation_id: path_3_fix_wave_r1_prime_prime_k_coin_pp_landed_20260526T125124Z
substrate_id: coin_pp_implicit_neural_representation
lane_id: lane_path_3_fix_wave_r1_prime_prime_k_coin_pp_memo_doctrine_baseline_20260526
fix_wave_round: R1_prime_prime_k
landed_utc: 2026-05-26T12:51:24Z
council_tier: T1
council_attendees: [Shannon, Carmack, Hotz, Quantizr]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
falsification_class: IMPLEMENTATION_LEVEL
related_deliberation_ids:
  - path_3_k_coin_pp_L0_scaffold_landed_20260526
  - path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526
  - path_3_recursive_adversarial_review_r1_prime_prime_aggregate_3_axis_landings_h_i_j_k_20260526
  - mlx_first_everywhere_canonical_doctrine_20260526
council_decisions_recorded:
  - "FIX-WAVE-R1''-K landed: 3-file correction + canonical equation registration"
  - "K landing memo APPEND-ONLY correction footer (Catalog #110/#113)"
  - "K test threshold updated to canonical floor classifier (5e-3 -> CANONICAL_ABS_MAX_UPPER_BOUND=6e-2)"
  - "MLX-first doctrine APPEND-ONLY canonical hardware-floor section"
  - "Canonical equation mlx_matmul_drift_m_series_canonical_floor_v1 registered (Catalog #344)"
  - "K counter 0/3 RESET unblocked; ready for fresh R1' counter on next review round"
horizon_class: frontier_pursuit
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
audit_evidence_tag: "[macOS-MLX research-signal]"
---

# Path 3 K — FIX-WAVE-R1''-K — COIN++ empirical-claim falsification closure + MLX-first canonical hardware floor

**Lane:** `lane_path_3_fix_wave_r1_prime_prime_k_coin_pp_memo_doctrine_baseline_20260526` L1
**Per R1'' verdict** (`.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md` per-substrate verdict NOT_CLEAN_FIX_WAVE_REQUIRED, counter RESET 0/3):
**This fix-wave closes** R1''-K-1 finding (5e-3 empirical anchor falsified) + lands canonical M-series hardware-floor baseline.

## 1. Scope (per operator brief)

5 deliverables landed in same commit batch:

| # | Deliverable | File | Surface |
|---|---|---|---|
| 1 | K landing memo APPEND-ONLY correction footer | `.omx/research/path_3_k_coin_pp_L0_scaffold_landed_20260526.md` | Memo correction per Catalog #110/#113 |
| 2 | K test threshold update (5e-3 -> canonical floor classifier) | `src/tac/substrates/coin_pp_implicit_neural_representation/tests/test_basic.py` | Test threshold expansion |
| 3 | MLX-first doctrine APPEND-ONLY canonical hardware-floor section | `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` | Doctrine baseline update |
| 4 | NEW canonical equation `mlx_matmul_drift_m_series_canonical_floor_v1` | `src/tac/canonical_equations/mlx_matmul_m_series_floor.py` + registry | Catalog #344 canonical-equation registration |
| 5 | THIS landing memo (PV evidence + 6-hook + sister coordination) | `.omx/research/path_3_fix_wave_r1_prime_prime_k_coin_pp_landed_*.md` | Landing memo |

## 2. Catalog #229 PV evidence

Read 4 canonical files BEFORE editing:

1. `.omx/research/path_3_k_coin_pp_L0_scaffold_landed_20260526.md` — K landing memo (215 lines; verbatim §3 Axis 2 line 116 + line 21 `council_decisions_recorded` empirical claim text)
2. `.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md` — R1'' per-substrate review (the canonical falsification source; §4.2 CRITICAL FINDING K-R1''-1)
3. `.omx/research/path_3_recursive_adversarial_review_r1_prime_prime_aggregate_3_axis_landings_h_i_j_k_20260526.md` — R1'' aggregate memo §8 Empirical anchor table (cross-validates H+K convergence on same hardware floor)
4. `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` — MLX-first doctrine (4107bbf8d; the canonical doctrine receiving the hardware-floor section)

Plus checked: `tac.canonical_equations` registry state (42 equations pre-landing, none for matmul-floor); `src/tac/canonical_equations/` package structure (12 files; sister `mlx_pytorch_drift.py` exists for sister downstream-scorer-drift; new sibling `mlx_matmul_m_series_floor.py` natural location); independent verification script run at `/tmp/r1_pp_k_drift_verify.py` confirming R1'' findings empirically.

## 3. Independent verification table

Reproduced R1'' findings on this M-series MPS machine using the same MLX↔numpy fp32 matmul measurement methodology. Per-dim measurements:

| (m,k)@(k,n)         | abs_max  | rms      | rel_median | rel_p95   |
|---------------------|----------|----------|------------|-----------|
| (32,32)@(32,32)     | 1.54e-2  | 4.52e-3  | 7.76e-4    | 3.77e-3   |
| (64,64)@(64,64)     | 2.42e-2  | 6.16e-3  | 7.66e-4    | 3.85e-3   |
| (128,128)@(128,128) | 3.62e-2  | 8.81e-3  | 7.59e-4    | 4.08e-3   |
| (256,64)@(64,256)   | 2.97e-2  | 6.20e-3  | 7.64e-4    | 3.87e-3   |
| (64,256)@(256,64)   | 4.60e-2  | 1.24e-2  | 7.75e-4    | 4.18e-3   |

Sinusoidal encoding (sin+cos at d=64, n=128): **abs_max = 1.19e-7 (bit-exact)**.

Test fixture (4x16)@(16x8) used by K's existing test: abs_max = 4.56e-3 (which IS within the original 5e-3 claim — the test passed in isolation, but the test fixture is smaller than substrate-typical dims and does NOT generalize as a canonical anchor).

COIN++ MOD_DIM=64 hidden=64 depth=3 chain (full topology, 384*512 pixels): abs_max = 2.81e-1 (compounding error across 3 sin-film-linear layers; outside the per-matmul floor — relevant for downstream scorer-drift sister equation per `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1`).

**Conclusion**: R1'' findings confirmed. Canonical M-series MPS fp32 matmul drift hardware floor is **O(1e-2) abs / O(1e-3) rms / 7.6e-4 rel-median** across K-typical dims; sinusoidal encoding is bit-exact special case. K's headline 5e-3 anchor was an artifact of measuring only the small test fixture.

## 4. Falsification classification per Catalog #307

**IMPLEMENTATION-LEVEL FALSIFICATION** of the empirical claim:

- The COIN++ substrate paradigm (meta-learned modulated INR per Dupont 2022) remains INTACT per R1'' §3 Axis 1 (10/10 HARD-EARNED layers; UNIQUE-AND-COMPLETE-PER-METHOD compliance)
- The substrate's architectural primitives (sinusoidal encoding / FiLM / coord-MLP / sigmoid output / brotli q=9 / fp16 state_dict / canonical helpers) all retain HARD-EARNED-CANONICAL status
- The substrate's anti-pattern avoidance (NO `mx.repeat` / NO `align_corners=True` / NO `mx.softmax` / NO non-Kahan large-N / NO fp16 matmul) IS empirically verified per R1'' §4.3
- Only the headline empirical-anchor NUMBER (5e-3 abs) was wrong; the architecture is not falsified

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "KILL/FALSIFIED memory verdicts" non-negotiables: K substrate is DEFERRED-pending-fix-wave-closure (NOT killed) + resumes at fresh R1' counter post-correction.

## 5. Per-deliverable diff summary

### 5.1 K landing memo APPEND-ONLY correction footer (Deliverable 1)

Added `## APPEND-ONLY CORRECTION FOOTER (FIX-WAVE-R1''-K landing 2026-05-26)` section after existing Cross-references. Per Catalog #110/#113 HISTORICAL_PROVENANCE: the original §3 Axis 2 line 116 + line 21 + frontmatter line 21 prose is preserved verbatim. The footer records:

- Falsification class (IMPLEMENTATION-LEVEL per Catalog #307)
- Verbatim falsified claim citation
- Canonical R1'' independent verification table (per-dim measurements with `[empirical:<artifact>]` tag per Catalog #287)
- Corrected canonical anchor (replaces line 116 + line 21 claim downstream)
- Test threshold update reference
- Canonical equation registration reference
- Counter status (0/3 RESET unblocked)
- Sister coordination at fix-wave landing time
- Cross-references

### 5.2 K test threshold update (Deliverable 2)

`src/tac/substrates/coin_pp_implicit_neural_representation/tests/test_basic.py::test_mlx_numpy_parity_skipped_if_mlx_unavailable`:

- Replaced hardcoded `assert max_abs < 5e-3` with canonical helper invocation:
  - Imports `classify_mlx_matmul_drift` + `CANONICAL_ABS_MAX_UPPER_BOUND` from new sister module
  - Computes both `max_abs` AND `rms` of drift (per canonical equation contract)
  - Asserts verdict in {`BIT_EXACT_LIKE_SINUSOIDAL`, `WITHIN_CANONICAL_FLOOR`} (covers the small-fixture bit-exact case AND the canonical floor band)
  - Error message cites canonical floor + canonical equation + R1''-K source for debug context
- Test docstring updated with CORRECTION pointer + reactivation criteria + cross-references to canonical equation + `[empirical:<artifact>]` tag per Catalog #287
- All 26 K tests still pass post-update (verified locally)

### 5.3 MLX-first doctrine APPEND-ONLY canonical hardware-floor section (Deliverable 3)

`.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md`: Added new section `## M-series MPS fp32 hardware floor canonical anchor (2026-05-26 R1'' verification)` after the existing `EOF` marker per Catalog #110/#113 APPEND-ONLY discipline. The original doctrine prose remains unchanged. The section:

- Cites canonical equation `mlx_matmul_drift_m_series_canonical_floor_v1` as the binding reference
- Documents canonical floor (abs_max upper bound 6e-2 / rms upper bound 1.5e-2 / rel_median floor 7.6e-4 / sinusoidal bit-exact 1.2e-7)
- Per-dim measurements table (5 substrate dims) with `[empirical:<artifact>]` per Catalog #287
- Canonical-substrate-design implication: substrates requiring <1e-2 abs MUST route through Kahan/fp64 mitigation OR accept PROXY-grade per Catalog #341 Tier A
- Producer/consumer map per Catalog #344
- Cross-substrate META implication (R1'' aggregate §8 H+K convergence anchors the hardware-class property)
- Reactivation criteria (per-M-series-class characterization for cross-machine canonical promotion)
- Sister convergence anchor (H + K independent verification convergence)

### 5.4 NEW canonical equation `mlx_matmul_drift_m_series_canonical_floor_v1` (Deliverable 4)

NEW module `src/tac/canonical_equations/mlx_matmul_m_series_floor.py` (~260 LOC) exposing:

- `EQUATION_ID = "mlx_matmul_drift_m_series_canonical_floor_v1"`
- Canonical constants: `CANONICAL_ABS_MAX_UPPER_BOUND=6e-2`, `CANONICAL_RMS_UPPER_BOUND=1.5e-2`, `CANONICAL_REL_MEDIAN=7.6e-4`, `CANONICAL_SINUSOIDAL_ENCODING_BIT_EXACT=1.2e-7`
- `classify_mlx_matmul_drift(measured_abs_max, measured_rms=None, measured_rel_median=None, matmul_shape=None)` canonical classifier
- `build_mlx_matmul_drift_m_series_canonical_floor_v1()` builder
- Verdict taxonomy: `BIT_EXACT_LIKE_SINUSOIDAL` / `WITHIN_CANONICAL_FLOOR` / `ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION`
- All canonical non-promotable markers per Catalog #127/#192/#317/#341 (`evidence_grade="macOS-MLX research-signal"` + `axis_tag="[macOS-MLX research-signal]"` + `score_claim=False` + `promotion_eligible=False` + etc.)

Registered via `register_canonical_equation` per Catalog #344. Registry event details:

- **Registration UTC**: 2026-05-26T12:45:00Z
- **Producers** (2): `path_3_fix_wave_r1_prime_prime_k_independent_verification`, `tools.measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift`
- **Consumers** (5): `tac.substrates.coin_pp_implicit_neural_representation.tests.test_basic`, `path_3_recursive_adversarial_review_r1_prime_prime_axis_2_reviewer`, `tools.gate_mlx_candidate_contest_equivalence`, `tac.cathedral_consumers.canonical_equation_lookup_consumer`, `mlx_first_everywhere_canonical_doctrine`
- **Initial empirical anchor**: `r1_pp_k_independent_verification_5_substrate_dims_20260526` (5-dim sweep; worst-case abs_max 4.60e-2; residual 0.767 = 4.60e-2 / 6.00e-2 upper bound)
- **Recalibration trigger**: `RECALIBRATE_ON_NEW_ANCHORS` (when 3+ new empirical anchors land — e.g. per-M-series-class characterization adds 3+ anchors then auto-recalibrate)
- **Domain of validity**: framework_pair=MLX vs numpy; dtype=fp32; hardware_substrate_class=darwin_arm64_apple_silicon_m_series_mps; matmul_dim_range=32-256; out_of_domain when matmul dim > 256 / fp16 matmul / non-M-series Apple Silicon / non-Apple-Silicon hardware; bit_exact_primitive_exception for sin/cos/sigmoid/elementwise

### 5.5 Sister tests verification

- K substrate test suite: `26 passed in 0.65s` (no regressions; new canonical-floor-based assertion replaces hardcoded 5e-3)
- Canonical equations test suite: `121 passed in 1.48s` (no regressions; new sibling module integrates cleanly)
- Catalog #344 STRICT preflight on new files: 0 violations (canonical equation registration cites canonical-equations module per the gate's required token list; new files do not add to pre-existing 110 WARN-ONLY baseline)

## 6. Counter advancement readiness

Per R1'' verdict: K=COIN++ per-substrate counter was **0/3 (RESET)** due to R1''-K-1 CRITICAL finding. With FIX-WAVE-R1''-K landing:

- ✅ R1'' finding K-R1''-1 (5e-3 anchor falsified) — CLOSED via Deliverable 1 (memo correction) + Deliverable 2 (test threshold update) + Deliverable 4 (canonical equation as canonical reference)
- ✅ Counter **0/3 RESET unblocked** — K is ready for fresh R1' counter on next review round
- ✅ All substrate code unchanged (only test threshold + memo); architecture + 3-axis HARD-EARNED layers + anti-pattern avoidance all preserved
- ✅ Per CLAUDE.md "Forbidden premature KILL without research exhaustion": K substrate remains DEFERRED-pending-fresh-R1' (NOT killed); paradigm INTACT

## 7. Sister-substrate META-FINDING

R1'' aggregate memo §8 measured EQUIVALENT drift on H (atw_v2_cooperative_receiver_v2) at the SAME hardware floor:

- H typical dim (64,256): abs=4.97e-2 (within canonical floor 6e-2)
- K typical dim (64,256)@(256,64): abs=4.60e-2 (within canonical floor 6e-2)

The H + K convergence empirically validates the canonical floor IS a **hardware-class property** (M-series Apple Silicon MPS fp32 matmul accumulation), not a per-substrate artifact.

**Sister-substrate cross-check candidate**: other Path 3 substrates whose landing memos claim empirical MLX drift values that should be cross-checked against R1'' canonical floor:

| Substrate | Landing memo drift claim | Recommended action |
|---|---|---|
| A=DreamerV3 (pre-FIX-WAVE-R1) | max_abs=24.34 (anti-pattern driven) | Already corrected in FIX-WAVE-R1 commit `e1b101888`; sister of K |
| F=Z8 (pre-FIX-WAVE-R1' G) | LINE-FOR-LINE inheritance of A bugs (3.77 + 1.51) | Already corrected via CONSOLIDATE-OP-1 extraction `caf29acdb`; sister of K |
| B'/C'/D/E/G | Various per-substrate drift claims | If any claim <1e-2 abs, cross-check against canonical floor |
| H=atw_v2_cooperative_receiver_v2 | "1e-3 to 1e-2 abs" band (per landing memo) | FIX-WAVE-R1''-H IN-FLIGHT (DISJOINT from this lane) |
| I=faiss_ivf_pq_residual | NO actual MLX primitives (substrate-package surface vacuous) | FIX-WAVE-R1''-I IN-FLIGHT (DISJOINT from this lane) |
| J=mdl_ibps_j_discrete_categorical_mine_hybrid | (clean per R1'' aggregate; advanced to 1/3) | No correction needed |

**Recommendation**: extend R1''-K canonical floor verification to remaining Path 3 substrates' empirical drift claims if any claim <1e-2 abs (operator-routable next-step).

## 8. 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = N/A at this fix-wave (canonical floor anchor itself is per-primitive; downstream consumers route through `tac.sensitivity_map.*`)
- hook #2 Pareto constraint = N/A (canonical floor is per-primitive drift bound, not Pareto-relevant)
- hook #3 bit-allocator = N/A (canonical floor does not affect per-byte allocation)
- hook #4 cathedral autopilot dispatch = ACTIVE (canonical equation auto-discovered by `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335 + Catalog #344)
- hook #5 continual-learning posterior = ACTIVE (canonical equation registered via `register_canonical_equation`; APPEND-ONLY `EVENT_REGISTERED` row landed in `.omx/state/canonical_equations_registry.jsonl`)
- hook #6 probe-disambiguator = ACTIVE (canonical equation `classify_mlx_matmul_drift` IS the disambiguator between BIT_EXACT_LIKE_SINUSOIDAL / WITHIN_CANONICAL_FLOOR / ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION verdicts)

## 9. Sister coordination (Catalog #230) — current state at landing

**IN-FLIGHT** sisters (verified DISJOINT scope):

- L2-LONGTRAIN-D-Z6 (`ac44cc1a555b14df4`) — touches `src/tac/substrates/time_traveler_l5_z6/`; DISJOINT
- FIX-WAVE-R1''-H (`a35f7d7a5601fcad7`) — touches `src/tac/substrates/atw_v2_cooperative_receiver_v2/`; DISJOINT
- FIX-WAVE-R1''-I (`ab03b57a92e45dc0e`) — touches `src/tac/substrates/faiss_ivf_pq_residual/`; DISJOINT

**THIS landing** touches (ZERO collision with H/I/L2-Z6):

- `src/tac/substrates/coin_pp_implicit_neural_representation/tests/test_basic.py` (K-only)
- `.omx/research/path_3_k_coin_pp_L0_scaffold_landed_20260526.md` (K memo APPEND-ONLY footer)
- `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` (APPEND-ONLY new section)
- `src/tac/canonical_equations/mlx_matmul_m_series_floor.py` (NEW canonical equation module)
- `.omx/state/canonical_equations_registry.jsonl` (one NEW `EVENT_REGISTERED` event via canonical helper; APPEND-ONLY per Catalog #131/#138)
- THIS landing memo (NEW file)

## 10. Discipline compliance

- ✅ Catalog #229 PV (read K landing memo + R1'' per-substrate memo + R1'' aggregate memo + MLX-first doctrine + canonical equations registry state BEFORE editing)
- ✅ Catalog #117/#157/#174 canonical serializer with POST-EDIT `--expected-content-sha256` (commit pending; per-file sha computed at commit time)
- ✅ Catalog #119 Co-Authored-By trailer (appended by canonical serializer)
- ✅ Catalog #110/#113 APPEND-ONLY (K landing memo correction footer + MLX-first doctrine new section + canonical equations registry NEW event — ZERO mutation of existing prose / events)
- ✅ Catalog #208 docs/local-paths (no `/Users/adpena/...` paths in any artifact)
- ✅ Catalog #230 sister-subagent ownership map (DISJOINT scope verified vs L2-LONGTRAIN-D-Z6 + FIX-WAVE-R1''-H + FIX-WAVE-R1''-I; THIS landing carries no overlap)
- ✅ Catalog #287 placeholder-rationale rejection (corrected anchor carries `[empirical:<artifact path>]` tag; no placeholder rationales)
- ✅ Catalog #307 IMPLEMENTATION-LEVEL classification (paradigm INTACT; only empirical claim wrong)
- ✅ Catalog #340 sister-checkpoint guard PROCEED (no file overlap with in-flight subagents)
- ✅ Catalog #344 canonical equation registered + APPEND-ONLY ledger event
- ✅ Per CLAUDE.md "MLX portable-local-substrate authority": empirical artifact tagged `[macOS-MLX research-signal]` + non-promotable markers per Catalog #341
- ✅ Per CLAUDE.md "MPS auth eval is NOISE": canonical equation IS hardware-FLOOR reference only — NEVER score authority
- ✅ Per CLAUDE.md "Executing actions with care": NO `gh pr create`, NO `gh release create`, NO Modal/Vast/Lightning dispatch
- ✅ Per CLAUDE.md "Apples-to-apples evidence discipline": independent verification reproduced R1'' findings on same machine; canonical floor anchor matches measurement methodology

## 11. Mission contribution per Catalog #300

`frontier_breaking_enabler` — extincts the empirical-anchor falsification bug class structurally at FOUR surfaces:

1. **Memo surface** (K landing memo correction footer + canonical citation)
2. **Test threshold surface** (test routes through canonical classifier; future regressions surface as verdict shift)
3. **Doctrine surface** (MLX-first doctrine canonical hardware-floor section as binding reference)
4. **Canonical equations registry surface** (canonical equation registered per Catalog #344; auto-discovered by cathedral consumers per Catalog #335)

The structural extinction means future Path 3 (and Path N) substrate designs:

- MUST cite the canonical equation in design memos per Catalog #344
- CAN use the canonical classifier in MLX parity tests (instead of hardcoding drift literals)
- CAN reason about per-matmul accuracy requirement at design-memo time per axis 2 discipline
- WILL inherit the canonical floor as binding reference per MLX-first doctrine

Sister of FIX-WAVE-R1''-H + FIX-WAVE-R1''-I (3-of-4 R1'' fix-waves) — together they close the R1'' wave + unblock R2'' synchronized counter-advance.

## 12. Operator-routable next-step

**Extend R1''-K canonical floor verification to remaining Path 3 substrates' empirical drift claims**: if any sister substrate's landing memo claims empirical drift at <O(1e-2) abs, cross-check against the canonical floor via:

```bash
# Inspect canonical floor + classify per-substrate measured drift
PYTHONPATH=src .venv/bin/python -c "
from tac.canonical_equations.mlx_matmul_m_series_floor import classify_mlx_matmul_drift
result = classify_mlx_matmul_drift(measured_abs_max=<sister_substrate_claim>)
print(result['verdict'])
"
```

Per-substrate cross-check candidates surfaced in §7 META-FINDING above. The canonical equation's `RECALIBRATE_ON_NEW_ANCHORS` trigger means 3+ new sister-substrate cross-validation anchors will auto-recalibrate the canonical floor — providing structural protection that the canonical hardware-floor anchor stays empirically grounded as more substrates land.

## 13. Cost + wall-clock

- **Paid GPU**: $0 (memo + canonical equation registration; pure $0 work)
- **Wall-clock**: ~60 min (within ~60 min estimate per operator brief)
- **Token efficiency** (per operator pacing directive #4): parallel reads (4 source files batched); independent empirical verification reproducible from /tmp/r1_pp_k_drift_verify.py + /tmp/r1_pp_k_drift_results.json

## 14. Cross-references

- R1'' per-substrate memo (canonical falsification source):
  `.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`
- R1'' aggregate memo §8 (independent verification + H+K convergence anchor):
  `.omx/research/path_3_recursive_adversarial_review_r1_prime_prime_aggregate_3_axis_landings_h_i_j_k_20260526.md`
- K landing memo (carries APPEND-ONLY correction footer):
  `.omx/research/path_3_k_coin_pp_L0_scaffold_landed_20260526.md`
- MLX-first canonical doctrine (carries APPEND-ONLY hardware-floor section):
  `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md`
- Canonical equation module: `src/tac/canonical_equations/mlx_matmul_m_series_floor.py`
- Canonical equations registry: `.omx/state/canonical_equations_registry.jsonl`
- K test (updated): `src/tac/substrates/coin_pp_implicit_neural_representation/tests/test_basic.py::test_mlx_numpy_parity_skipped_if_mlx_unavailable`
- Sister canonical equation: `tac.canonical_equations.mlx_pytorch_drift.build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1`
- CONSOLIDATE-OP-1 canonical MLX primitives: commit `caf29acdb`
- CLAUDE.md "MLX portable-local-substrate authority" + "MPS auth eval is NOISE" + "Apples-to-apples evidence discipline" + "Forbidden premature KILL without research exhaustion" + "Canonical equations + models registry"
- Catalog #110 / #113 / #117 / #119 / #127 / #157 / #174 / #185 / #186 / #192 / #206 / #208 / #229 / #230 / #287 / #299 / #307 / #317 / #335 / #340 / #341 / #344

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
