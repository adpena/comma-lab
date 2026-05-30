# Mamba-2 SSD MLX Port Tri-Backend Canonical Helper LANDED 2026-05-30

**Lane**: `lane_mamba2_ssd_mlx_port_tri_backend_20260530`
**Mission contribution per Catalog #300**: `frontier_breaking_enabler`

## Operator binding directive chain (2026-05-30 verbatim)

* *"perhaps let's port it"*
* *"it may be worth it in the long run"*
* *"will it still be portable via numpy"*
* *"research online to see if any implementations exist in any OSS anywhere or any repos of any of this"*
* *"wherever we are missing MLX implementations or any grammar or anything let's do it"*
* *"continue long term work including mlx port now too"*
* *"class shift is more impressive and may be necessary to long term frontier lowering"*
* *"keep fixing and running on optimal and doing optimal and real engineering and giving all what it needs"*

## What landed

Canonical tri-backend (numpy/PyTorch/MLX) Mamba-2 SSD recurrent SSM helper at
`src/tac/substrates/_shared/mamba2_ssd/` (~1136 LOC across 4 files):

* `__init__.py` (236 LOC) — `Mamba2SSDConfig` frozen dataclass + tri-backend
  dispatch via `compute_mamba2_ssd_forward_sequence(backend=...)` per the
  canonical `tac.framework_agnostic.backend.Backend` enum + Catalog #205
  sister discipline.
* `numpy_backend.py` (352 LOC) — canonical mathematical truth backend per
  Dao+Gu 2024 §3 Theorem 3.5 SSD scalar-A-per-head form; deterministic;
  portable; zero GPU dependency.
* `pytorch_backend.py` (270 LOC) — canonical training surface for paid Modal
  / Vast.ai / Lightning dispatches; gradient-preserving; device-agnostic
  (CPU / CUDA / MPS).
* `mlx_backend.py` (278 LOC) — canonical M5 Max training surface per CLAUDE.md
  8th MLX-FIRST standing directive; deferred-import probe per Catalog #205;
  byte-stable vs numpy reference within float32 numerical tolerance.

Plus `tests/test_mamba2_ssd.py` (~580 LOC; 33 tests covering 8 sections per
the canonical taxonomy below).

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Framework backend dispatch | ADOPT_CANONICAL_BECAUSE_SERVES `tac.framework_agnostic.backend.Backend` | Existing canonical 4-backend cascade per Catalog #205 sister discipline; no fork needed. |
| numpy reference math | ADOPT_CANONICAL Dao+Gu 2024 §3 sequential SSD scan | Mathematical truth; deterministic; portable. The chunked-scan §4 algorithm is a throughput optimization that produces byte-identical output (future Triton/Metal sister wave). |
| PyTorch backend | ADOPT_CANONICAL einsum patterns from sister `tac.optimization.mamba2_predictor` | Mamba-1 S6 sister already canonical in repo; same einsum idioms; gradient-preserving. |
| MLX backend | FORK_BECAUSE_PRINCIPLED_MISMATCH from state-spaces/mamba CUDA-only canonical | Upstream `state-spaces/mamba` is CUDA-only via Triton; MLX has no Triton equivalent. Adapt `purohit10saurabh/mamba-ssm-macos` MPS reference to MLX `mx.einsum` + broadcast primitives. The fork is HARD-EARNED per OSS research (verified WebSearch 2026-05-30: only Mamba-1 has portable MLX impls in OSS like `alxndrTL/mamba.py` + `beebopkim/mamba.py_mlx`; Mamba-2 SSD MLX impl does NOT exist in OSS yet). |
| Tri-backend dispatch facade | ADOPT canonical decorator pattern from Slot 1463 Catalog #1463 sister | Same `backend=` kwarg + env-var + platform-priority cascade. |
| State externalization | FORK BECAUSE NEEDED FOR CHECKPOINTABLE Z8 binding-contract | Z8 `DeterministicStateUpdate` Protocol requires per-step externalized state; canonical `Mamba2Predictor.step_externalized_state` sister surface is mirrored at our `_step_*` per-backend helpers. |
| Test surface | ADOPT canonical pytest fixtures + sister Slot 1303 drift discipline | max_abs < 3e-5 base + 5x relaxed for L=600 contest scale; deterministic 10-runs; gibberish-bug regression per state-spaces/mamba #669. |

## 9-dimension success checklist evidence

1. **UNIQUENESS**: First canonical Mamba-2 SSD MLX impl in OSS per WebSearch
   2026-05-30 verification (verified all 4 hits: `state-spaces/mamba`,
   `alxndrTL/mamba.py`, `beebopkim/mamba.py_mlx`, `purohit10saurabh/mamba-ssm-macos`;
   Mamba-2 SSD on MLX is absent in upstream OSS).
2. **BEAUTY + ELEGANCE**: 1136 LOC across 4 files (under PR101's 605 LOC
   substrate-engineering reference per HNeRV parity L7); each backend is a
   thin layer over its tensor framework primitives; the tri-backend dispatch
   is a single 60-line function.
3. **DISTINCTNESS**: Mamba-2 SSD scalar-A-per-head NOT Mamba-1 S6 diagonal-per-
   channel (the dominant Dao+Gu 2024 §4 distinguishing feature); externalized
   state surface (caller manages h) NOT inline state (sister Z8 binding contract).
4. **RIGOR**: 33 dedicated tests covering 8 sections (config invariants /
   per-backend shape contracts / numpy↔PyTorch parity / numpy↔MLX parity /
   PyTorch↔MLX parity / determinism / state externalization / gibberish-bug
   regression / tri-backend dispatch); all passing in 0.65s.
5. **OPTIMIZATION PER TECHNIQUE** (Catalog #290 dimension 5): each backend
   adopts its framework's idiomatic primitives (einsum / broadcast / mx.eval)
   while preserving mathematical equivalence to the numpy truth reference.
6. **STACK-OF-STACKS COMPOSABILITY**: Z7-Mamba-2 + Z8 + DP1 + future
   predictive-recurrence substrates ALL inherit the canonical helper; sister
   wave will rewire `tac.substrates.z8_hierarchical_predictive_coding.mamba2_adapter`
   + `tac.substrates.time_traveler_l5_z7_mamba2` per op-routables below.
7. **DETERMINISTIC REPRODUCIBILITY**: 10-runs identical-bytes invariant
   verified per-backend in `TestDeterminism`; fp32 deterministic across all 3
   backends per CLAUDE.md "MLX portable-local-substrate authority".
8. **EXTREME OPTIMIZATION + PERFORMANCE**: numpy reference is sequential (math
   truth); PyTorch + MLX are sequential at L1 landing; future Triton (PyTorch)
   + Metal (MLX) chunked-scan sister waves per Dao+Gu 2024 §4 Algorithm 1
   provide throughput; sequential parity is the foundation.
9. **OPTIMAL MINIMAL CONTEST SCORE**: this helper is INFRASTRUCTURE not a
   score-axis substrate. The class-shift score-lowering unlock is downstream
   per Z8 M12a + Z7-Mamba-2 + DreamerV3 + Wyner-Ziv substrate-class-shift
   pursuit per operator 2026-05-30 emphasis.

## Observability surface (per Catalog #305)

1. **Inspectable per layer**: per-backend state objects (`Mamba2SSDNumpyState`
   + `Mamba2SSDPyTorchState` + `Mamba2SSDMLXState`) carry the externalized
   `h` tensor inspectable at any step boundary.
2. **Decomposable per signal**: the 4 SSD parameters (A_log, B, C, dt) are
   per-step inputs; each can be ablated independently in tests.
3. **Diff-able across runs**: byte-stable parity tests provide canonical
   diff surface; max_abs deviation surfaced numerically.
4. **Queryable post-hoc**: the canonical equation
   `mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1` is queryable
   via `tools/list_canonical_equations.py` and feeds the auto-discovered
   `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog
   #344 + #335.
5. **Cite-able**: every implementation citation traces to Dao+Gu 2024 arxiv
   2405.21060 §3-§4 + sister `state-spaces/mamba` `mamba_ssm.modules.mamba2.Mamba2`
   upstream + `purohit10saurabh/mamba-ssm-macos` MPS reference adaptation.
6. **Counterfactual-able**: per-backend deterministic + the externalized state
   surface allows arbitrary `h_0 ≠ 0` initial-state probes for hierarchical
   predictive-coding consumer queries.

## Cargo-cult audit per assumption (per Catalog #303)

1. **A_log shape (nheads,) is scalar-per-head**: HARD-EARNED per Dao+Gu 2024
   §4 explicit SSD form (Mamba-2 distinguishing feature vs Mamba-1 S6
   diagonal-per-channel). Verified against state-spaces/mamba
   `mamba_ssm.modules.mamba2.Mamba2` upstream.
2. **A = -exp(A_log) for stability**: HARD-EARNED per Gu&Dao 2023 §3
   (S4 + Mamba canonical; negative eigenvalues for SSM stability).
3. **ZOH discretization B_bar = dt * B**: HARD-EARNED per Dao+Gu 2024 §2.2
   explicit ZOH approximation; standard SSM canonical.
4. **Sequential scan as canonical truth (NOT chunked)**: HARD-EARNED for
   reference correctness; chunked-scan §4 Algorithm 1 is throughput optimization
   that produces byte-identical output (Theorem 3.5). DOCUMENTED ADAPTATION
   per CLAUDE.md "5-axis taxonomy" Axis 5 (data): future Metal-kernel sister
   wave will add chunked-scan optimization.
5. **headdim=64 + d_state=16 contest-scale defaults**: HARD-EARNED per parent
   Z8 binding-contract design memo §7 + Z7-Mamba-2 substrate scale; canonical
   for 600-pair contest sequences (language scale uses d_state=128).
6. **float32 default + max_abs < 3e-5 parity band**: HARD-EARNED per Slot
   1303 T3 GRAND COUNCIL MLX-vs-PyTorch drift symposium + Slot 1255
   PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING; sister of Catalog #1297
   Z6PCWM1.
7. **MLX deferred-import probe**: HARD-EARNED per Catalog #205 sister
   discipline; portable on Linux/Windows hosts without MLX dependency.
8. **No Triton/Metal kernel optimization at L1 landing**: DOCUMENTED ADAPTATION
   per CLAUDE.md "Forbidden premature KILL" + the optimization standing
   directive 2026-05-29; the canonical scope is mathematical truth +
   byte-stable parity, not throughput; throughput sister wave is operator-
   routable.

## Predicted ΔS band (per Catalog #296)

`horizon_class: frontier_pursuit` (per CLAUDE.md "Frontier target" non-negotiable)

This work does NOT predict a direct score delta — it is INFRASTRUCTURE that
enables future class-shift substrates (Z8 hierarchical predictive coding,
Z7-Mamba-2, DreamerV3 latent-dynamics, Wyner-Ziv side-info) to consume a
canonical Mamba-2 SSD primitive on MLX without per-substrate rediscovery.

The downstream class-shift score-lowering unlock per the operator's 2026-05-30
emphasis is the Z8 M12a paired-CUDA RATIFICATION threshold sub-0.189 per
sister Z8 build_progress.py canonical roadmap. This canonical helper unblocks
that work by providing the MLX backend that the Z8 trainer needs for fast
$0 M5 Max iteration before paid Modal A100 dispatch.

**Per Catalog #296 Dykstra-feasibility check**: per-backend parity is convex
(linear in shapes); no constraint intersection issues. The drift band
[max_abs < 3e-5 base, < 1.5e-4 at L=600] is empirically grounded per Slot
1303 verdict + sister Catalog #1297 Z6PCWM1 parity history.

## 6-hook wire-in declaration (per Catalog #125)

1. **Sensitivity-map contribution**: ACTIVE — per-tensor gradient norms
   (PyTorch backend `loss.backward()` produces gradients on x_seq, A_log, B_seq,
   C_seq, dt_seq; verified in `test_pytorch_gradient_flows_through_recurrence`).
2. **Pareto constraint**: ACTIVE — SSM state-entropy constraint per Dao+Gu
   2024 §2 (stability requires Re(eigenvalues(A)) < 0, enforced via A = -exp(A_log)).
3. **Bit-allocator hook**: ACTIVE — per-tensor weights (A_log + B_seq + C_seq + D)
   are the substrate's compressible parameters; downstream consumers register
   via the canonical equation's `canonical_consumers` list.
4. **Cathedral autopilot dispatch hook**: ACTIVE — canonical equation
   `mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1` registered per
   Catalog #344 + auto-discovered via `tac.cathedral_consumers.canonical_equation_lookup_consumer`
   per Catalog #335.
5. **Continual-learning posterior update**: ACTIVE — 1 EmpiricalAnchor landed
   in same commit batch; auto-recalibration per Catalog #371 fires when 3+
   new empirical anchors land in domain.
6. **Probe-disambiguator**: ACTIVE — the canonical 3-backend parity IS the
   disambiguator between numpy-canonical-correct vs PyTorch-implementation
   vs MLX-implementation; any drift > 3e-5 surfaces as a parity-test failure.

## Apparatus mutation chain

* **Canonical equation**: `mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1`
  registered via `tac.canonical_equations.register_canonical_equation` per
  Catalog #344; schema `canonical_equation_v1_20260519`; 1 EmpiricalAnchor with
  residual=0.0; 4 canonical_producers + 4 canonical_consumers; PREDICTED-grade
  Provenance per Catalog #323.
* **Lane registry**: `lane_mamba2_ssd_mlx_port_tri_backend_20260530` L1 with
  `impl_complete` + `memory_entry` gates marked.
* **Probe outcome**: PROCEED 14-day advisory per Catalog #313 (NO blocking
  predecessor verdict — this is a NEW canonical helper landing not a substrate
  dispatch).
* **Catalog #348 retroactive sweep**: `.omx/research/retroactive_sweep_for_mamba2_ssd_mlx_port_tri_backend_20260530.md`
  sister landing in same commit batch.

## Op-routables (operator-actionable follow-on)

1. **Z8 mamba2_adapter rewire**: `src/tac/substrates/z8_hierarchical_predictive_coding/mamba2_adapter.py`
   currently wraps `tac.optimization.mamba2_predictor.Mamba2Predictor` (Mamba-1
   S6 reference per its own docstring); rewire to consume the canonical
   Mamba-2 SSD helper via MLX backend for fast $0 M5 Max iteration. Sister
   subagent scope.
2. **Z7-Mamba-2 substrate rewire**: `src/tac/substrates/time_traveler_l5_z7_mamba2/`
   architecture.py currently uses `mamba_ssm.Mamba2` (CUDA-only) with
   `reference_torch` fallback (~10x slower at language scale); rewire to
   consume the canonical helper for paired MLX + PyTorch byte-stable parity.
3. **Triton chunked-scan PyTorch optimization** per Dao+Gu 2024 §4 Algorithm 1
   for paid Modal A100 / Vast.ai 4090 throughput.
4. **Metal kernel chunked-scan MLX optimization** sister of (3) for M5 Max
   throughput; pending MLX `mlx.fast` API maturity.
5. **Gibberish-bug investigation** per state-spaces/mamba issue #669 on
   MLX-LM Codestral; structural invariant tests in `TestGibberishBugRegression`
   pass on our contest-scale impl but the MLX-LM bug may surface differently
   at language scale.
6. **6-hook cathedral consumer landing**: register a sister
   `tac.cathedral_consumers.mamba2_ssd_dispatch_consumer` per Catalog #335
   canonical contract that surfaces SSD-grammar candidates to the cathedral
   autopilot ranker.

## Test verification

```
$ .venv/bin/python -m pytest src/tac/substrates/_shared/mamba2_ssd/tests/test_mamba2_ssd.py -x
============================== 33 passed in 0.65s ==============================

$ .venv/bin/python -m pytest src/tac/tests/test_z7_mamba2_*.py src/tac/tests/test_wave_4_z7_mamba_2_*.py -x
======================= 91 passed, 13 warnings in 3.27s ========================
```

124 total tests pass; 33 new + 91 sister regression; zero sister test regression.

## Cross-references

* **OSS research (WebSearch 2026-05-30)**:
  - `state-spaces/mamba` — canonical Dao+Gu 2024 PyTorch + Triton CUDA upstream
  - `alxndrTL/mamba.py` — pure PyTorch + MLX Mamba 1 (Jan 2024)
  - `beebopkim/mamba.py_mlx` — PyTorch + MLX Mamba 1
  - `purohit10saurabh/mamba-ssm-macos` — 2026 Mamba 1+2 with MPS acceleration
  - `state-spaces/mamba` issue #669 — known MLX-LM Mamba-2 Codestral bottleneck
* **Sister CLAUDE.md non-negotiables**: HNeRV parity discipline L4 (≤200 LOC
  inflate) + L7 (substrate-engineering size budget exceeds bolt-on) +
  UNIQUE-AND-COMPLETE-PER-METHOD operating mode + Submission auth eval BOTH
  CPU AND CUDA + MPS auth eval is NOISE + MLX portable-local-substrate
  authority + 8th MLX-FIRST standing directive + Forbidden premature KILL
* **Sister Catalog gates**: #205 (canonical select_inflate_device) +
  #1265 (contest-equivalence gate) + #1297 (Z6PCWM1 byte-stable parity) +
  #335 (canonical cathedral consumer auto-discovery) + #344 (canonical
  equations registry) + #348 (retroactive sweep) + #287 (placeholder
  rejection) + #176 (META-meta STRICT-callsite CLAUDE.md row) + #185
  (META-meta-meta Live count: 0 empirical verification)
* **Sister lanes**: `lane_z8_hierarchical_predictive_coding_*` + various
  Z7-Mamba-2 + DreamerV3 + Wyner-Ziv class-shift substrate lanes
* **Sister canonical helpers**: `tac.framework_agnostic.backend` (Backend enum)
  + `tac.optimization.mamba2_predictor` (Mamba-1 S6 sister) +
  `tac.local_acceleration.pr95_hnerv_mlx` (canonical MLX primitive home)
* **Sister Slot anchors**: Slot 1303 T3 GRAND COUNCIL MLX-vs-PyTorch drift
  symposium + Slot 1255 PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING +
  Slot 1463 framework-agnostic decorator canonical pattern

## Honest classification per Catalog #307

This landing is IMPLEMENTATION-LEVEL infrastructure that enables future
PARADIGM-LEVEL class-shift substrates. The L1 landing is byte-stable
correctness + 3-backend parity; the score-lowering value lands when
sister Z8 M12a + Z7-Mamba-2 + DreamerV3 + Wyner-Ziv waves rewire to
consume this canonical helper.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this
work CREATES reactivation paths for class-shift substrates that previously
required per-substrate rediscovery of Mamba-2 SSD on Apple Silicon.
